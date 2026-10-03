#!/usr/bin/env python
"""
重新生成 fig:city_level (逐城市 R² + RMSE 双 y 轴单图)。

- R² / RMSE 使用官方 20 种子逐城市均值，取自 error_analysis.per_city_errors
  (trainer 在【全部 3 天预测窗口】上算出的权威指标，与论文 tab:results 口径一致)。
- 左轴 R² (蓝)，右轴 RMSE (橙)，x 轴为英文城市名 (按 R² 降序)。
- 样式: 纯白背景、黑色坐标轴 (左/下/右)、图例左上角。

输出: result/paper_figures_20seed/visualizations/prediction/city_level_Full_KAN_Graphormer.{png,svg,pdf}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_city_level.py
"""
import sys, os, pickle, argparse
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
from exp_lib.visualizer import ImprovedActualDataVisualizer

CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']

BASE = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
N_SEEDS = 20


def load_per_city_errors(seed):
    pkl_path = os.path.join(BASE, f'seed_{seed}', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result['eval_data']['error_analysis']['per_city_errors']


def per_city_metrics_20seed():
    """逐城市 R² / RMSE: 20 种子均值，来自官方 error_analysis.per_city_errors."""
    r2s = np.zeros((N_SEEDS, len(CITIES)))
    rmses = np.zeros((N_SEEDS, len(CITIES)))
    for s in range(N_SEEDS):
        pce = load_per_city_errors(s)
        for ci, cn in enumerate(CITIES):
            r2s[s, ci] = float(pce[cn]['r2'])
            rmses[s, ci] = float(pce[cn]['rmse'])
    return r2s.mean(axis=0), r2s.std(axis=0, ddof=1), rmses.mean(axis=0), rmses.std(axis=0, ddof=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=r'E:\Claude code\KAN+\result\paper_figures_20seed')
    args = parser.parse_args()

    mean_r2, std_r2, mean_rmse, std_rmse = per_city_metrics_20seed()
    print('逐城市 20 种子官方指标 (mean ± std):')
    for ci, cn in enumerate(CITIES):
        print(f'  {cn:6s}  R² = {mean_r2[ci]:+.3f} ± {std_r2[ci]:.3f}   '
              f'RMSE = {mean_rmse[ci]:.3f} ± {std_rmse[ci]:.3f}')

    # 传入 seed_0 的 eval_results 仅用于满足函数前置检查 (数值会被 override 覆盖)
    pkl_path = os.path.join(BASE, 'seed_0', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl_path, 'rb') as f:
        er = pickle.load(f)['eval_data']['evaluation_results']
    os.makedirs(args.out, exist_ok=True)
    viz = ImprovedActualDataVisualizer(args.out, CITIES)
    viz._plot_city_level_predictions(er, 'Full_KAN_Graphormer',
                                     city_r2_override=mean_r2,
                                     city_rmse_override=mean_rmse,
                                     city_r2_std=std_r2,
                                     city_rmse_std=std_rmse)
    print('Done. 输出目录:', os.path.join(args.out, 'visualizations', 'prediction'))


if __name__ == '__main__':
    main()
