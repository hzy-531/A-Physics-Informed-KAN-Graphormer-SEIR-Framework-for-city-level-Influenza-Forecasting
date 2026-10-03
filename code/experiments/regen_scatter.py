#!/usr/bin/env python
"""
重新生成 fig:scatter (预测 vs 真实散点图)。

- 点来自 seed_0 的 Full_KAN_Graphormer 测试集预测 (26×24)。
- 标注使用 20 种子的均值±标准差 (与 tab:results 一致)。
- 样式: 纯白背景、黑色方框坐标轴 (上下左右)。

输出: result/paper_figures_20seed/prediction/scatter_Full_KAN_Graphormer.{png,svg,pdf}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_scatter.py
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

from exp_lib.visualizer import ImprovedActualDataVisualizer

# 20 种子 Full KAN-G 的均值 ± 标准误 (与 tab:results 完全一致)
METRICS_20SEED = (0.744, 0.002, 3.503, 0.015, 1.901, 0.012)

CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pkl', type=str, default=None)
    parser.add_argument('--out', type=str, default=r'E:\Claude code\KAN+\result\paper_figures_20seed')
    args = parser.parse_args()

    if args.pkl:
        pkl_path = args.pkl
    else:
        base = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
        pkl_path = os.path.join(base, f'seed_{args.seed}', 'results',
                                'Full_KAN_Graphormer_result.pkl')

    print(f'Pkl: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)

    eval_results = result['eval_data']['evaluation_results']
    os.makedirs(args.out, exist_ok=True)
    viz = ImprovedActualDataVisualizer(args.out, CITIES)
    viz._plot_prediction_scatter(eval_results, 'Full_KAN_Graphormer',
                                 metrics_override=METRICS_20SEED)
    print('Done. 输出目录:', os.path.join(args.out, 'prediction'))


if __name__ == '__main__':
    main()
