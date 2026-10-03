#!/usr/bin/env python
"""
重新生成 fig:lockdown (封城参数对比图)，使用 20 种子运行的 Full 结果 (seed 0)。

输出: result/paper_figures_20seed/visualizations/comparison/lockdown_params_comparison.{png,svg,pdf}

用法:
    PYTHONIOENCODING=utf-8 python experiments/regen_lockdown.py
    PYTHONIOENCODING=utf-8 python experiments/regen_lockdown.py --seed 0
    PYTHONIOENCODING=utf-8 python experiments/regen_lockdown.py --pkl <路径>
"""
import sys, os, pickle, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 旧 pkl 中的类定义在 __main__ 模块中，需注册兼容类
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

    if 'full_pred' not in result:
        fp_path = os.path.join(os.path.dirname(pkl_path), 'full_pred.pkl')
        if os.path.exists(fp_path):
            with open(fp_path, 'rb') as f:
                result['full_pred'] = pickle.load(f)

    cities = list(result.get('full_pred', {}).get('cities', [])) or [
        '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
        '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
        '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']

    os.makedirs(args.out, exist_ok=True)
    viz = ImprovedActualDataVisualizer(args.out, cities)
    viz._plot_lockdown_params_comparison({'Full_KAN_Graphormer': result})
    print('Done. 输出目录:', os.path.join(args.out, 'visualizations', 'comparison'))


if __name__ == '__main__':
    main()
