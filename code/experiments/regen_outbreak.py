#!/usr/bin/env python
"""
重新生成 fig:outbreak_merged (Guangzhou / Wuhan / Shanghai 爆发概率鲁棒性图)。

相对旧的 plot_outbreak_merged_panels.py 的改进:
- 数据改用当前 20 种子 (pred_total 逐元素求 20 种子均值, true_total 取 seed_0)。
- 三张子图【整合为一个坐标系】: hspace=0 无间距堆叠, 共用一根 x 轴 (仅底部显示日期),
  内部不画横向分隔线, 只保留最外层一个黑框 (上/下/左/右各一条连续黑边)。
- 纯白背景, 去掉网格线。
- 纵轴按城市单独摆放: 左轴 = 每日新增病例 (各城尺度不同), 右轴 = 爆发概率 (0~1, 三城同尺度)。
- 城市名 (英文) 写在子图内左上角, 图例放在整图顶部单行。

输出: result/paper_figures_20seed/visualizations/prediction/outbreak_robustness_3panels.{pdf,svg,png}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_outbreak.py
"""
import sys, os, pickle
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42      # 矢量 PDF, 放大不糊
matplotlib.rcParams['svg.fonttype'] = 'none'

BASE = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
OUT_DIR = r'E:\Claude code\KAN+\result\paper_figures_20seed\visualizations\prediction'
N_SEEDS = 20

TARGET_CITIES = ['广州市', '武汉市', '上海市']
CITY_EN = {'广州市': 'Guangzhou', '武汉市': 'Wuhan', '上海市': 'Shanghai'}
PANEL_LETTERS = ['a', 'b', 'c']

EPS = 1.0
K_MC = 200
PAST_WINDOW = 7
PRED_WINDOW = 3
LOOKBACK = 7
CONTEXT_BEFORE = 10

# 三组检测阈值: 宽松 / 基准 / 严格
THRESHOLD_CONFIGS = [
    {'I_min': 20, 'rho': 1.10, 'label': r'$I_{\min}=20,\ \rho=1.10$',
     'color': '#ff7f0e', 'linestyle': '--', 'alpha': 0.18},
    {'I_min': 30, 'rho': 1.25, 'label': r'$I_{\min}=30,\ \rho=1.25$',
     'color': '#17becf', 'linestyle': '-',  'alpha': 0.25},
    {'I_min': 50, 'rho': 1.50, 'label': r'$I_{\min}=50,\ \rho=1.50$',
     'color': '#9467bd', 'linestyle': '-.', 'alpha': 0.18},
]

INCIDENCE_COLOR = '#1f77b4'


def load_20seed():
    """pred_total 取 20 种子均值, true_total/dates 取 seed_0 (真值各种子一致)。"""
    preds = []
    true = dates = None
    for seed in range(N_SEEDS):
        pkl = os.path.join(BASE, f'seed_{seed}', 'results', 'full_pred.pkl')
        with open(pkl, 'rb') as f:
            fp = pickle.load(f)
        preds.append(np.array(fp['pred_total']))
        if seed == 0:
            true = np.array(fp['true_total'])
            dates = list(fp['dates'])
    pred = np.mean(np.stack(preds, axis=0), axis=0)
    return pred, true, dates


def compute_outbreak_probability(pred_total, true_total, sigma_city, city_idx,
                                  I_min, rho, lookback=7, predict=3):
    n_days = pred_total.shape[0]
    p_arr = pred_total[:, city_idx]
    t_arr = true_total[:, city_idx]
    sigma = sigma_city
    pct = np.full(n_days, np.nan)
    ci_lo = np.full(n_days, np.nan)
    ci_up = np.full(n_days, np.nan)
    label_arr = np.full(n_days, np.nan)

    for t in range(lookback, n_days):
        win = min(PRED_WINDOW, n_days - t)   # 尾部未来窗口不足 3 天时缩短, 补全到最后一天
        past = np.mean(t_arr[t - PAST_WINDOW:t])
        future = np.mean(t_arr[t:t + win])
        R_label = (future + EPS) / (past + EPS)
        label_arr[t] = 1.0 if (future >= I_min and R_label >= rho) else 0.0

        future_pred = np.mean(p_arr[t:t + win])
        mc_future = np.random.normal(future_pred, sigma / np.sqrt(win), size=K_MC)
        past_pred = np.mean(p_arr[t - PAST_WINDOW:t])
        mc_R = (mc_future + EPS) / (past_pred + EPS)
        Z_k = (mc_future >= I_min) & (mc_R >= rho)
        p = np.mean(Z_k)
        pct[t] = p

        z = 1.96
        d = 1 + z**2 / K_MC
        center = (p + z**2 / (2 * K_MC)) / d
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * K_MC)) / K_MC) / d
        ci_lo[t] = max(0.0, center - margin)
        ci_up[t] = min(1.0, center + margin)

    return pct, ci_lo, ci_up, label_arr


def style_unified_panel(ax1, ax2, is_top, is_bottom):
    """把三块面板拼成一个坐标系: 只保留最外层黑框, 内部无横向分隔线。

    - ax1 (左轴): 左脊保留(连续左边), 右脊关闭(twinx 负责右边),
      上脊仅顶部面板, 下脊仅底部面板。
    - ax2 (右轴): 右脊保留(连续右边), 左/上/下脊关闭。
    """
    ax1.set_facecolor('white')
    ax2.set_facecolor('none')
    # ax1 左脊
    ax1.spines['left'].set_visible(True)
    ax1.spines['left'].set_color('black')
    ax1.spines['left'].set_linewidth(0.8)
    # ax1 右脊关闭
    ax1.spines['right'].set_visible(False)
    # ax1 上/下脊: 仅外层
    ax1.spines['top'].set_visible(is_top)
    ax1.spines['bottom'].set_visible(is_bottom)
    for s in ('top', 'bottom'):
        if ax1.spines[s].get_visible():
            ax1.spines[s].set_color('black')
            ax1.spines[s].set_linewidth(0.8)
    # ax2 右脊
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(0.8)
    # ax2 左/上/下关闭
    for s in ('left', 'top', 'bottom'):
        ax2.spines[s].set_visible(False)


def main():
    pred, true, dates = load_20seed()
    n_days = pred.shape[0]
    print(f'数据: {n_days}天 × 24城 (pred 取 {N_SEEDS} 种子均值)')

    cities = ['北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
              '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
              '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']

    total_samples = n_days - LOOKBACK - PRED_WINDOW + 1
    train_size = int(total_samples * 0.6)
    train_end = LOOKBACK + train_size
    x_full = np.arange(n_days)

    # 预计算各城概率/CI/爆发标签, 统一 x 范围取最早爆发起点
    per_city = {}
    all_ob_starts = []
    for city_name in TARGET_CITIES:
        idx = cities.index(city_name)
        vt_true = true[train_end:n_days - PRED_WINDOW, idx]
        vt_pred = pred[train_end:n_days - PRED_WINDOW, idx]
        sigma_vt = float(np.std(vt_pred - vt_true))
        results = []
        for cfg in THRESHOLD_CONFIGS:
            pct, ci_lo, ci_up, label_arr = compute_outbreak_probability(
                pred, true, sigma_vt, idx, cfg['I_min'], cfg['rho'], LOOKBACK, PRED_WINDOW)
            results.append({'pct': pct, 'ci_lo': ci_lo, 'ci_up': ci_up,
                            'label_arr': label_arr, 'cfg': cfg})
        ob_days = np.where(results[1]['label_arr'] == 1.0)[0]
        if len(ob_days) > 0:
            all_ob_starts.append(int(ob_days[0]))
        per_city[city_name] = {'results': results, 'idx': idx}

    # x 范围: 从最早爆发起点前推 CONTEXT_BEFORE 天, 到 12-06 (与旧图一致, 只画早期预警窗口)
    xlim_start = max(0, min(all_ob_starts) - CONTEXT_BEFORE) if all_ob_starts else 0
    xlim_end = next((i for i, d in enumerate(dates) if d.month == 12 and d.day == 6), n_days - 1)
    mask = (x_full >= xlim_start) & (x_full <= xlim_end)
    x_vis = x_full[mask]
    print(f'x 范围: day {xlim_start} ({dates[xlim_start].strftime("%Y-%m-%d")}) ~ '
          f'day {xlim_end} ({dates[xlim_end].strftime("%Y-%m-%d")}), 爆发起点={all_ob_starts}')

    # 每 5 天一个刻度 (与旧图一致)
    tick_days = [d for d in range(xlim_start, xlim_end, 5)] + [xlim_end]
    tick_days = sorted(set(tick_days))
    tick_labels = [dates[d].strftime('%m-%d') for d in tick_days]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    twins = []

    for i, (city_name, ax1) in enumerate(zip(TARGET_CITIES, axes)):
        idx = per_city[city_name]['idx']
        results = per_city[city_name]['results']

        # 左轴: 每日新增病例 (散点, 不用连线)
        ax1.scatter(x_vis, true[mask, idx], s=16, marker='o', color=INCIDENCE_COLOR,
                    alpha=0.7, edgecolors='white', linewidth=0.4,
                    label='Daily new cases')
        ax1.set_ylabel('Daily New Cases', fontsize=12, color=INCIDENCE_COLOR)
        ax1.tick_params(axis='y', labelcolor=INCIDENCE_COLOR, labelsize=10)
        ax1.set_xlim(xlim_start, xlim_end)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # 右轴: 爆发概率 (三城同尺度 0~1)
        ax2 = ax1.twinx()
        twins.append(ax2)
        for res in results:
            cfg = res['cfg']
            ax2.fill_between(x_vis, res['ci_lo'][mask], res['ci_up'][mask],
                             alpha=cfg['alpha'], color=cfg['color'], linewidth=0.0)
            ax2.plot(x_vis, res['pct'][mask], cfg['linestyle'], color=cfg['color'],
                     linewidth=1.6, label=cfg['label'])
        ax2.set_ylabel('Outbreak Probability', fontsize=12)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.set_ylim(0, 1.15)

        # 子图标题: 面板字母 + 城市名 (英文), 放左上角
        ax1.set_title(f'({PANEL_LETTERS[i]}) {CITY_EN[city_name]}',
                      fontsize=12, fontweight='bold', loc='left')

    # 底部共用一根 x 轴: 仅最下面面板显示日期刻度 (旋转 45° 避免重叠)
    axes[-1].set_xticks(tick_days)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    # 图例: 左轴散点 + 三条阈值曲线, 放整图顶部单行
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = twins[0].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.5, 0.94), fontsize=10, ncol=4, framealpha=0.95)

    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        path = os.path.join(OUT_DIR, f'outbreak_robustness_3panels.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Done. 输出目录:', OUT_DIR)


if __name__ == '__main__':
    main()
