#!/usr/bin/env python
"""
将 Guangzhou / Wuhan / Shanghai 三城的爆发概率鲁棒性图合并为一张大图 (3 个子图，纵向堆叠)。

每个子图 = 一个城市:
  左轴: True Incidence (每日新增病例，散点图，不用连线——导师要求)
  右轴: 3 条不同阈值的 P_outbreak 曲线 + CI (保留多阈值版本)

三城用同一套配色 (发病率蓝色散点 + 橙/红/紫三条阈值曲线)，城市名在子图标题区分。

用法:
    PYTHONIOENCODING=utf-8 python experiments/plot_outbreak_merged_panels.py
    PYTHONIOENCODING=utf-8 python experiments/plot_outbreak_merged_panels.py --dir <result_dir>
"""
import sys, os, pickle, argparse
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'  # 可编辑文本的 SVG

CITY_PINYIN = {'广州市': 'guangzhou', '武汉市': 'wuhan', '上海市': 'shanghai'}
TARGET_CITIES = ['广州市', '武汉市', '上海市']

EPS = 1.0
K_MC = 200
PAST_WINDOW = 7
PRED_WINDOW = 3
CONTEXT_BEFORE = 10

# 三组阈值: 宽松 / 基准 / 严格 (保留多阈值)
THRESHOLD_CONFIGS = [
    {'I_min': 20, 'rho': 1.10, 'label': r'$y_{\min}=20,\ \rho=1.10$',
     'color': '#ff7f0e', 'linestyle': '--', 'alpha': 0.18},
    {'I_min': 30, 'rho': 1.25, 'label': r'$y_{\min}=30,\ \rho=1.25$',
     'color': '#d62728', 'linestyle': '-',  'alpha': 0.25},
    {'I_min': 50, 'rho': 1.50, 'label': r'$y_{\min}=50,\ \rho=1.50$',
     'color': '#9467bd', 'linestyle': '-.', 'alpha': 0.18},
]


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

    for t in range(lookback, n_days - predict):
        past = np.mean(t_arr[t - PAST_WINDOW:t])
        future = np.mean(t_arr[t:t + PRED_WINDOW])
        R_label = (future + EPS) / (past + EPS)
        label_arr[t] = 1.0 if (future >= I_min and R_label >= rho) else 0.0

        future_pred = np.mean(p_arr[t:t + PRED_WINDOW])
        mc_future = np.random.normal(future_pred, sigma / np.sqrt(PRED_WINDOW), size=K_MC)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()

    if args.dir:
        result_dir = args.dir
    else:
        base = r'E:\Claude code\KAN+\result'
        subdirs = sorted([d for d in os.listdir(base)
                          if os.path.isdir(os.path.join(base, d)) and d.startswith('ablation')],
                         reverse=True)
        if not subdirs:
            print('No result dir found.'); return
        result_dir = os.path.join(base, subdirs[0])

    print(f'Result dir: {result_dir}')
    fp_path = os.path.join(result_dir, 'results', 'full_pred.pkl')
    with open(fp_path, 'rb') as f:
        data = pickle.load(f)

    true = np.array(data['true_total'])
    pred = np.array(data['pred_total'])
    dates_list = data['dates']
    cities = ['北京市','天津市','上海市','重庆市','广州市','深圳市','西安市','成都市',
              '武汉市','杭州市','南京市','苏州市','无锡市','郑州市','长沙市','沈阳市',
              '大连市','青岛市','济南市','宁波市','厦门市','哈尔滨市','长春市','石家庄市']

    lookback, predict = 7, 3
    n_days = len(dates_list)
    total_samples = n_days - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    train_end = lookback + train_size
    x_full = np.arange(n_days)
    date_labels = [d.strftime('%m-%d') if i % 5 == 0 else '' for i, d in enumerate(dates_list)]

    # 右边界固定 12 月 6 日 (与单城图一致)
    xlim_end = next((i for i, d in enumerate(dates_list)
                     if d.month == 12 and d.day == 6), n_days - 1)

    # 预计算各城结果, 统一 x 范围取三城最早的爆发起点
    per_city = {}
    all_ob_starts = []
    for city_name in TARGET_CITIES:
        idx = cities.index(city_name)
        vt_true = true[train_end:n_days - predict, idx]
        vt_pred = pred[train_end:n_days - predict, idx]
        sigma_vt = float(np.std(vt_pred - vt_true))

        results = []
        for cfg in THRESHOLD_CONFIGS:
            pct, ci_lo, ci_up, label_arr = compute_outbreak_probability(
                pred, true, sigma_vt, idx, cfg['I_min'], cfg['rho'], lookback, predict)
            results.append({'pct': pct, 'ci_lo': ci_lo, 'ci_up': ci_up,
                            'label_arr': label_arr, 'cfg': cfg})

        base_label = results[1]['label_arr']
        ob_days = np.where(base_label == 1.0)[0]
        if len(ob_days) > 0:
            all_ob_starts.append(int(ob_days[0]))
        per_city[city_name] = {'results': results, 'idx': idx}

    xlim_start = max(0, min(all_ob_starts) - CONTEXT_BEFORE) if all_ob_starts else 0
    mask = (x_full >= xlim_start) & (x_full <= xlim_end)
    x_vis = x_full[mask]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    twins = []
    panel_letters = ['a', 'b', 'c']

    for i, (city_name, ax1) in enumerate(zip(TARGET_CITIES, axes)):
        idx = per_city[city_name]['idx']
        results = per_city[city_name]['results']

        # 左轴: True Incidence (散点图, 不用连线)
        ax1.scatter(x_vis, true[mask, idx], s=16, marker='o', color='#1f77b4',
                    alpha=0.7, edgecolors='white', linewidth=0.4,
                    label='Daily new cases')
        ax1.set_ylabel('Daily New Cases', fontsize=12, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=10)
        ax1.set_xlim(xlim_start, xlim_end)
        ax1.grid(True, linestyle=':', alpha=0.5)

        # 右轴: 3 条阈值曲线 + CI
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

        ax1.set_title(f'({panel_letters[i]}) {CITY_PINYIN[city_name].capitalize()}',
                      fontsize=12, fontweight='bold', loc='left')

    # X 轴刻度 (sharex 下仅底部子图显示标签)
    axes[-1].set_xticks(x_vis)
    axes[-1].set_xticklabels([date_labels[i] for i in range(n_days) if mask[i]],
                             rotation=45, ha='right', fontsize=9)

    # 图例 (取第一个子图的双轴句柄, 放图顶部)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = twins[0].get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               bbox_to_anchor=(0.5, 0.94), fontsize=10, ncol=4, framealpha=0.95)

    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_dir = os.path.join(result_dir, 'visualizations', 'prediction')
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'outbreak_robustness_3panels.png')
    pdf_path = os.path.join(output_dir, 'outbreak_robustness_3panels.pdf')
    svg_path = os.path.join(output_dir, 'outbreak_robustness_3panels.svg')
    plt.savefig(png_path, dpi=250, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=250, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()

    print(f'xlim=[{xlim_start},{xlim_end}]  outbreak_starts={all_ob_starts}')
    print(f'  OK -> {png_path}')
    print(f'  OK -> {pdf_path}')
    print(f'  OK -> {svg_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
