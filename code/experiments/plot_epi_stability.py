#!/usr/bin/env python
"""
对比 Full KAN-G 与 M-Graphormer 的流行病学参数随时间的演化 (β / γ⁻¹ / C / Rt)，
用 20 种子画 mean ± SE，用阴影带展示跨种子的稳定性 (带越窄 = 越稳定)。

    Full KAN-G   -> 原 20 种子结果 (ablation_20seeds_6models_20260915_200323)
    M-Graphormer -> 容量对齐 20 种子结果 (ablation_20seeds_align_20260919_003254)

两者 hidden_dim=32, weight_decay=1e-2，与 tab:epi 口径一致 (公平对比)。

输出: result/paper_figures_20seed/visualizations/comparison/epi_stability_full_vs_mgraph.{pdf,png}

用法:
    PYTHONIOENCODING=utf-8 python experiments/plot_epi_stability.py
"""
import sys, os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 旧 pkl 中的类定义在 __main__ 模块，需注册兼容类才能 pickle.load
import __main__
from exp_lib import models, layers, physics, loss, data

_COMPAT_CLASSES = [
    ('EnhancedFull_Graphormer_V3', models.EnhancedFull_Graphormer_V3),
    ('EnhancedMLP_Baseline_V3', models.EnhancedMLP_Baseline_V3),
    ('EnhancedLSTM_Baseline_V3', models.EnhancedLSTM_Baseline_V3),
    ('EnhancedGAT_Baseline_V3', models.EnhancedGAT_Baseline_V3),
    ('EnhancedKAN_Only_V3', models.EnhancedKAN_Only_V3),
    ('M_Graphormer_Baseline', models.M_Graphormer_Baseline),
    ('NoPhysics_KAN_Graphormer', models.NoPhysics_KAN_Graphormer),
    ('RobustKANLinear', layers.RobustKANLinear),
    ('StandardMLP', layers.StandardMLP),
    ('GatedTCN', layers.GatedTCN),
    ('RBFSpatialEncoding', layers.RBFSpatialEncoding),
    ('GraphormerLayer', layers.GraphormerLayer),
    ('ImprovedGATLayer', layers.ImprovedGATLayer),
    ('DataCollector', data.DataCollector),
    ('CityLevelLogMinMaxScaler', data.CityLevelLogMinMaxScaler),
    ('CityLevelLinearMinMaxScaler', data.CityLevelLinearMinMaxScaler),
    ('TemporalFeatureEnhancer', data.TemporalFeatureEnhancer),
    ('ImprovedTemporalDataset', data.ImprovedTemporalDataset),
    ('CurriculumScientificLoss', loss.CurriculumScientificLoss),
]
for _name, _cls in _COMPAT_CLASSES:
    setattr(__main__, _name, _cls)

FULL_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
MGRAPH_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_align_20260919_003254'
N_SEEDS = 20


def to_np(t):
    if t is None:
        return None
    if isinstance(t, list):
        if len(t) == 0:
            return None
        t = np.concatenate([x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
                            for x in t], axis=0)
    elif hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    elif hasattr(t, 'numpy'):
        t = t.numpy()
    return np.array(t)


def load_all_seeds(result_dir, model, key, n_seeds=N_SEEDS):
    """读取 n_seeds 个种子的某个参数, 返回 (n_seeds, T, C)。"""
    series = []
    for s in range(n_seeds):
        pkl = os.path.join(result_dir, f'seed_{s}', 'results', f'{model}_result.pkl')
        with open(pkl, 'rb') as f:
            ed = pickle.load(f)
        arr = to_np(ed['eval_data']['evaluation_results'].get(key))
        series.append(arr)
    return np.stack(series, axis=0)


def get_dates():
    """从 Full seed_0 的 full_pred 里拿日期 (136 天), 拿不到则按起止日期生成。"""
    pkl = os.path.join(FULL_DIR, 'seed_0', 'results', 'Full_KAN_Graphormer_result.pkl')
    try:
        with open(pkl, 'rb') as f:
            ed = pickle.load(f)
        dates = ed.get('full_pred', {}).get('dates')
        if dates is not None and len(dates) > 0:
            return list(dates)
    except Exception:
        pass
    # 兜底: 2019-11-01 ~ 2020-03-15, 共 136 天
    return list(pd.date_range('2019-11-01', periods=136, freq='D'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str,
                        default=r'E:\Claude code\KAN+\result\paper_figures_20seed')
    args = parser.parse_args()

    # 每个参数: 城市维度求平均 -> (n_seeds, T)
    full = {}
    mgraph = {}
    for key in ['betas', 'gammas', 'contacts', 'r0s']:
        f = load_all_seeds(FULL_DIR, 'Full_KAN_Graphormer', key)      # (20, T, 24)
        m = load_all_seeds(MGRAPH_DIR, 'M_Graphormer_Baseline', key)
        if key == 'gammas':
            # γ⁻¹ = 逐 cell 取 1/γ 再对城市平均 (与 tab:epi 口径一致)
            f = 1.0 / f
            m = 1.0 / m
        full[key] = f.mean(axis=2)      # (20, T)
        mgraph[key] = m.mean(axis=2)

    T = full['betas'].shape[1]
    n_test = T

    # 测试期日期 (预测窗口起点 = train+val+lookback)
    dates = get_dates()
    lookback, predict = 7, 3
    total_samples = 136 - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    pred_start_day = train_size + val_size + lookback
    x_dates = dates[pred_start_day: pred_start_day + n_test]

    def mean_se(arr):
        mean = arr.mean(axis=0)                       # (T,)
        se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        return mean, se

    panels = [
        ('betas',    r'Transmission rate $\beta$',      r'$\beta$ (day$^{-1}$)'),
        ('gammas',   r'Infectious period $\gamma^{-1}$', r'$\gamma^{-1}$ (days)'),
        ('contacts', r'Contact coefficient $c$',        r'$c$'),
        ('r0s',      r'Effective reproduction $R_t$',   r'$R_t$'),
    ]

    plt.rcParams.update({
        'font.sans-serif': ['DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    x = np.arange(n_test)

    C_FULL, C_MG = '#1f77b4', '#ff7f0e'

    # 记录每个参数的平均 SE 宽度 (用于量化稳定性)
    print('=' * 66)
    print('跨种子稳定性对比 (mean ± SE, 20 seeds), SE 均值越小越稳定:')
    print('=' * 66)

    panel_letters = ['(a)', '(b)', '(c)', '(d)']
    for ax, letter, (key, title, ylabel) in zip(axes.flat, panel_letters, panels):
        f_mean, f_se = mean_se(full[key])
        m_mean, m_se = mean_se(mgraph[key])

        ax.plot(x, f_mean, '-', color=C_FULL, lw=2.0, label='KAN-Graphormer-SEIR')
        ax.fill_between(x, f_mean - f_se, f_mean + f_se,
                        color=C_FULL, alpha=0.22, lw=0)
        ax.plot(x, m_mean, '-', color=C_MG, lw=1.6, label='MLP-Graphormer-SEIR')
        ax.fill_between(x, m_mean - m_se, m_mean + m_se,
                        color=C_MG, alpha=0.22, lw=0)

        # 传染病学阈值参考线
        if key == 'r0s':
            ax.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.7)
            ax.text(0.5, 1.0, r'$R_t=1$', transform=ax.get_yaxis_transform(),
                    fontsize=9, color='gray', va='bottom')

        # x 轴日期 (mm-dd)
        n_xticks = 6
        xtick_pos = np.linspace(0, n_test - 1, n_xticks, dtype=int)
        ax.set_xticks(x[xtick_pos])
        ax.set_xticklabels([x_dates[i].strftime('%m-%d') for i in xtick_pos],
                           fontsize=9)
        ax.set_xlabel('Test date (2020)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_facecolor('white')
        for spine in ('top', 'right', 'left', 'bottom'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.0)

        # 子图编号 (a)(b)(c)(d)
        ax.text(0.02, 0.96, letter, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

        print(f'  {title:34s}  Full SE={f_se.mean():.4f}   '
              f'M-Graph SE={m_se.mean():.4f}   '
              f'比值(M/F)={m_se.mean() / max(f_se.mean(), 1e-9):.2f}x')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.00))

    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_dir = os.path.join(args.out, 'visualizations', 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, 'epi_stability_full_vs_mgraph')
    fig.savefig(stem + '.pdf', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(stem + '.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('=' * 66)
    print(f'图已保存: {stem}.pdf / .png')


if __name__ == '__main__':
    import pandas as pd
    main()
