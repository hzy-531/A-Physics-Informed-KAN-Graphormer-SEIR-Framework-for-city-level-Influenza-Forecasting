#!/usr/bin/env python
"""
重新生成 fig:evolution (24 城参数演化大图, 6行×4列)。

改进点 (相对旧的 plot_city_evolution.py):
- 数据改用当前 20 种子 (seed_0) 的 full_pred，不再读 6 月旧目录。
- 输出 PDF(矢量)+SVG+PNG，矢量图放大不糊 (旧版只有 dpi=200 的 PNG)。
- 字号全面加大: 城市名 10 / 刻度 8 / 图例 8 (旧版 5~6.5)。
- 纯白背景、去掉网格; 坐标轴用黑色直线做成完整方框 (上下左右)。
- 刻度加粗: x 轴 5 个日期刻度, y 轴约 4 个。
- 城市名用英文放在子图内左上角; 图例放在子图内右上角。

输出: result/paper_figures_20seed/visualizations/evolution/evolution_24city_grid.{pdf,svg,png}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_evolution.py
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42      # PDF 字体为矢量 (可选中/放大不糊)
matplotlib.rcParams['svg.fonttype'] = 'none'  # SVG 字体保持为文本

CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']

_CITY_EN = {
    '北京市': 'Beijing', '天津市': 'Tianjin', '上海市': 'Shanghai',
    '重庆市': 'Chongqing', '广州市': 'Guangzhou', '深圳市': 'Shenzhen',
    '西安市': "Xi'an", '成都市': 'Chengdu', '武汉市': 'Wuhan',
    '杭州市': 'Hangzhou', '南京市': 'Nanjing', '苏州市': 'Suzhou',
    '无锡市': 'Wuxi', '郑州市': 'Zhengzhou', '长沙市': 'Changsha',
    '沈阳市': 'Shenyang', '大连市': 'Dalian', '青岛市': 'Qingdao',
    '济南市': 'Jinan', '宁波市': 'Ningbo', '厦门市': 'Xiamen',
    '哈尔滨市': 'Harbin', '长春市': 'Changchun', '石家庄市': 'Shijiazhuang',
}

BASE = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
OUT_DIR = r'E:\Claude code\KAN+\result\paper_figures_20seed\visualizations\evolution'
LOCKDOWN_DAY = 83  # 2020-01-23
TRIM_START = 7      # 截掉前端 lookback 学习期 (day 0–6 为 0)
TRIM_END = 133      # 截掉尾端未预测的几天 (day 134–135 为 0)

# 颜色 (与旧版一致, 保持论文 caption 的紫色封城线约定)
BETA_COLOR = '#1f77b4'      # β 蓝
CONTACT_COLOR = '#9467bd'   # C 紫
RT_COLOR = '#ff7f0e'        # Rt 橙
LOCKDOWN_COLOR = '#7B1FA2'  # 封城线 深紫
RT1_COLOR = '#999999'       # Rt=1 参考线 灰


def load_full_pred_mean(n_seeds=20):
    """加载 20 个种子的 full_pred，逐元素求均值 (β/C/Rt 的 20 种子平均轨迹)。"""
    betas, contacts, rts = [], [], []
    dates = None
    for seed in range(n_seeds):
        pkl = os.path.join(BASE, f'seed_{seed}', 'results', 'Full_KAN_Graphormer_result.pkl')
        with open(pkl, 'rb') as f:
            fp = pickle.load(f)['full_pred']
        betas.append(np.array(fp['beta_full']))
        contacts.append(np.array(fp['contact_full']))
        rts.append(np.array(fp['rt_full']))
        if dates is None:
            dates = list(fp['dates'])
    beta = np.mean(np.stack(betas, axis=0), axis=0)
    contact = np.mean(np.stack(contacts, axis=0), axis=0)
    rt = np.mean(np.stack(rts, axis=0), axis=0)
    return beta, contact, rt, dates


def style_box(ax, ax2):
    """纯白背景 + 黑色方框 (左/下/上来自 ax, 右来自 ax2)。"""
    ax.set_facecolor('white')
    ax2.set_facecolor('none')
    for spine in ('top', 'left', 'bottom'):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0.8)
    ax.spines['right'].set_visible(False)
    for spine in ('top', 'left', 'bottom'):
        ax2.spines[spine].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(0.8)


def main():
    beta, contact, rt, dates = load_full_pred_mean(20)
    n_days, n_cities = beta.shape
    print(f'数据: {n_days}天 × {n_cities}城 (20 种子均值)，截取 day {TRIM_START}–{TRIM_END}')

    # 截取中间非零段 (去掉前端 lookback 学习期和尾端未预测的几天)
    beta = beta[TRIM_START:TRIM_END + 1]
    contact = contact[TRIM_START:TRIM_END + 1]
    rt = rt[TRIM_START:TRIM_END + 1]

    # x 轴刻度: ~每月一个常规刻度 + 封城日特殊刻度
    regular_ticks = [TRIM_START, 37, 68, 99, TRIM_END]
    lockdown_str = dates[LOCKDOWN_DAY].strftime('%m/%d') if hasattr(dates[LOCKDOWN_DAY], 'strftime') else '01/23'
    all_ticks = sorted(set(regular_ticks + [LOCKDOWN_DAY]))
    all_labels = [(dates[i].strftime('%m/%d') if hasattr(dates[i], 'strftime') else str(dates[i])[:5])
                  for i in all_ticks]

    fig, axes = plt.subplots(6, 4, figsize=(16, 21))
    x = np.arange(TRIM_START, TRIM_END + 1)

    for idx, (city, ax) in enumerate(zip(CITIES, axes.flat)):
        b = beta[:, idx]
        c = contact[:, idx]
        r = rt[:, idx]

        # 左轴: β + Contact
        ax.plot(x, b, '-', color=BETA_COLOR, linewidth=1.3, label=r'$\beta$', alpha=0.95)
        ax.plot(x, c, '--', color=CONTACT_COLOR, linewidth=1.3, label='c', alpha=0.95)
        ax.set_ylabel(r'$\beta$ / c', fontsize=10, color=BETA_COLOR, labelpad=2)
        ax.tick_params(axis='y', labelsize=9, colors=BETA_COLOR)
        lo, hi = min(b.min(), c.min()), max(b.max(), c.max())
        span = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
        ax.yaxis.set_major_locator(MaxNLocator(4))

        # 右轴: Rt
        ax2 = ax.twinx()
        ax2.plot(x, r, '-', color=RT_COLOR, linewidth=1.5, label=r'$R_t$', alpha=0.95)
        ax2.set_ylabel(r'$R_t$', fontsize=10, color=RT_COLOR, labelpad=2)
        ax2.tick_params(axis='y', labelsize=9, colors=RT_COLOR)
        ax2.axhline(y=1.0, color=RT1_COLOR, linestyle='--', linewidth=0.7, alpha=0.7, zorder=1)
        rt_lo, rt_hi = r.min(), r.max()
        rt_span = max(rt_hi - rt_lo, 1e-6)
        ax2.set_ylim(rt_lo - 0.12 * rt_span, rt_hi + 0.12 * rt_span)
        ax2.yaxis.set_major_locator(MaxNLocator(4))

        # x 轴: 常规刻度 + 封城日紫色刻度 (不再画竖直封城线)
        ax.set_xticks(all_ticks)
        ax.set_xticklabels(all_labels, fontsize=9)
        for tl in ax.get_xticklabels():
            if tl.get_text() == lockdown_str:
                tl.set_color(LOCKDOWN_COLOR)
                tl.set_fontweight('bold')

        # 方框样式
        style_box(ax, ax2)

        # 城市名 (英文) 放子图内左上角
        ax2.text(0.02, 0.97, _CITY_EN[city], transform=ax2.transAxes,
                 fontsize=11, fontweight='bold', color='black', va='top', ha='left', zorder=20)

        # 图例只放第一个子图内右上角 (24 子图图例完全相同)
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper right', fontsize=9, framealpha=0.85, ncol=3,
                       handlelength=1.6, columnspacing=0.8, borderaxespad=0.3)

    # 24 城刚好铺满 6×4=24, 无需隐藏
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=1.0, h_pad=1.2, w_pad=1.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        path = os.path.join(OUT_DIR, f'evolution_24city_grid.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Done. 输出目录:', OUT_DIR)


if __name__ == '__main__':
    main()
