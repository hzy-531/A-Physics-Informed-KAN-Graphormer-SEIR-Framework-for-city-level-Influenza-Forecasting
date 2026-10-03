#!/usr/bin/env python
"""
重新生成 fig:timeseries (6 城市时序预测子图)。

- 曲线来自 seed_0 的 full_pred (trainer 权威的全序列预测): 取测试段最后 26 天。
  true_total / pred_total 均为 (136, 24)，测试段 = 索引 110:136 (2020-02-19 ~ 2020-03-15)。
- 每个子图标题用英文城市名，R² 用官方 20 种子逐城市均值 (来自
  error_analysis.per_city_errors，与论文 tab:results 口径一致)。
- 样式: 纯白背景、黑色单坐标轴 (左+下)。

输出: result/paper_figures_20seed/visualizations/prediction/timeseries_Full_KAN_Graphormer.{png,svg,pdf}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_timeseries.py
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
TEST_DAYS = 26  # 测试集长度 (20% of 136)


def load_result(seed):
    pkl_path = os.path.join(BASE, f'seed_{seed}', 'results', 'Full_KAN_Graphormer_result.pkl')
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def per_city_r2_20seed():
    """逐城市 R²: 官方 20 种子均值 (来自 error_analysis.per_city_errors)."""
    r2s = np.zeros((N_SEEDS, len(CITIES)))
    for s in range(N_SEEDS):
        pce = load_result(s)['eval_data']['error_analysis']['per_city_errors']
        for ci, cn in enumerate(CITIES):
            r2s[s, ci] = float(pce[cn]['r2'])
    return r2s.mean(axis=0), r2s.std(axis=0)


def load_full_pred_test(seed):
    """返回 full_pred 的测试段 (最后 TEST_DAYS 天): (pred, true)，均为 (TEST_DAYS, 24)."""
    fp = load_result(seed)['full_pred']
    pred_total = np.array(fp['pred_total'])
    true_total = np.array(fp['true_total'])
    return pred_total[-TEST_DAYS:], true_total[-TEST_DAYS:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default=r'E:\Claude code\KAN+\result\paper_figures_20seed')
    args = parser.parse_args()

    mean_r2, std_r2 = per_city_r2_20seed()
    print('逐城市 20 种子官方 R² (mean ± std):')
    for ci, cn in enumerate(CITIES):
        print(f'  {cn:6s}  R² = {mean_r2[ci]:+.3f} ± {std_r2[ci]:.3f}')

    pred_test, true_test = load_full_pred_test(args.seed)
    print(f'测试段形状: pred={pred_test.shape}, true={true_test.shape}')

    # 以 full_pred 测试段构造 results dict (满足 _plot_prediction_timeseries 的输入约定)
    results = {'predictions': pred_test, 'targets': true_test}
    os.makedirs(args.out, exist_ok=True)
    viz = ImprovedActualDataVisualizer(args.out, CITIES)
    viz._plot_prediction_timeseries(results, 'Full_KAN_Graphormer', city_r2_override=mean_r2)
    print('Done. 输出目录:', os.path.join(args.out, 'visualizations', 'prediction'))


if __name__ == '__main__':
    main()
