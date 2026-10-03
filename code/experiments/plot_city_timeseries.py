#!/usr/bin/env python
"""
各城市新增病例时间序列图 — 24城真实值 vs 预测值
自动读取最新 full_pred.pkl
"""
import sys, os, glob, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CITIES = ['北京市', '天津市', '石家庄市', '沈阳市', '哈尔滨市', '长春市',
          '上海市', '南京市', '杭州市', '合肥市', '武汉市', '长沙市',
          '广州市', '深圳市', '福州市', '南宁市', '成都市', '重庆市',
          '贵阳市', '昆明市', '西安市', '兰州市', '青岛市', '郑州市']

LOCKDOWN_DAY = 83  # 2020-01-23


def find_latest_pkl():
    dirs = sorted(glob.glob(r'E:\Claude code\KAN+\result\*'))
    for d in reversed(dirs):
        p = os.path.join(d, 'results', 'full_pred.pkl')
        if os.path.exists(p):
            return p
    return None


def plot_all_cities(pkl_path=None, output_dir=None):
    if pkl_path is None:
        pkl_path = find_latest_pkl()
    if pkl_path is None:
        print("❌ 找不到 full_pred.pkl")
        return

    print(f"读取: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    true = data['true_total']  # (136, 24)
    pred = data['pred_total']  # (136, 24)
    dates = data['dates']
    n_days, n_cities = true.shape

    # 计算各城市 R²
    city_r2 = {}
    for i in range(n_cities):
        t, p = true[:, i], pred[:, i]
        r2 = r2_score(t, p) if np.var(t) > 1e-5 else 0.0
        city_r2[i] = r2

    # 把日期转为 matplotlib 可用的格式
    date_objs = [d.to_pydatetime() if hasattr(d, 'to_pydatetime')
                 else pd.Timestamp(d).to_pydatetime() for d in dates]

    lockdown_date = date_objs[LOCKDOWN_DAY] if LOCKDOWN_DAY < len(date_objs) else None

    # 6行 × 4列
    fig, axes = plt.subplots(6, 4, figsize=(24, 28))
    fig.suptitle('24-City Daily New Cases: Actual vs Predicted (Full KAN-Graphormer)',
                 fontsize=18, fontweight='bold', y=0.995)

    for idx, (ax, city) in enumerate(zip(axes.flat, CITIES)):
        t_vals = true[:, idx]
        p_vals = pred[:, idx]
        r2 = city_r2[idx]

        ax.plot(date_objs, t_vals, 'b-', linewidth=1.5, alpha=0.85, label='Actual')
        ax.plot(date_objs, p_vals, 'r--', linewidth=1.2, alpha=0.75, label='Predicted')

        # 封城线
        if lockdown_date:
            ax.axvline(x=lockdown_date, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

        ax.set_title(f'{city}  R²={r2:.3f}', fontsize=10, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    # 去掉多余的子图
    for idx in range(n_cities, 24):
        axes.flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(pkl_path))
    out_path = os.path.join(output_dir, 'all_cities_timeseries.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存: {out_path}")

    # 打印 R² 排名
    ranked = sorted(city_r2.items(), key=lambda x: x[1], reverse=True)
    print("\n各城市 R² 排名:")
    for rank, (ci, r2) in enumerate(ranked, 1):
        flag = '⭐' if r2 > 0.8 else '✅' if r2 > 0.7 else '🟡' if r2 > 0.5 else '🔴'
        print(f"  {rank:2d}. {CITIES[ci]:<6}  R²={r2:.4f}  {flag}")

    # 总体统计
    all_true = true.flatten()
    all_pred = pred.flatten()
    global_r2 = r2_score(all_true, all_pred)
    global_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
    global_mae = np.mean(np.abs(all_pred - all_true))
    print(f"\n全局: R²={global_r2:.4f}  RMSE={global_rmse:.4f}  MAE={global_mae:.4f}")

    # 封城前后
    pre_mask = np.array([i < LOCKDOWN_DAY for i in range(n_days)])
    post_mask = np.array([i >= LOCKDOWN_DAY for i in range(n_days)])
    pre_r2 = r2_score(true[pre_mask].flatten(), pred[pre_mask].flatten())
    post_r2 = r2_score(true[post_mask].flatten(), pred[post_mask].flatten())
    print(f"封城前 R²={pre_r2:.4f}  封城后 R²={post_r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='24城新增病例时间序列图')
    parser.add_argument('--pkl', type=str, default=None, help='full_pred.pkl 路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    args = parser.parse_args()
    plot_all_cities(pkl_path=args.pkl, output_dir=args.output)
