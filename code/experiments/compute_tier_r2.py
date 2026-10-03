#!/usr/bin/env python
"""临时：计算 tab:spatial 分层 pooled R² (20 种子 mean ± std)。"""
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
from sklearn.metrics import r2_score

CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']

# 分层定义 (与 tab:spatial 的 Note 一致)
TIER1 = ['北京市', '上海市', '广州市', '深圳市']          # 4 城
TIER2 = ['天津市', '重庆市', '成都市', '武汉市', '南京市', '杭州市', '西安市', '郑州市']  # 8 城

BASE = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
N_SEEDS = 20


def load(seed):
    p = os.path.join(BASE, f'seed_{seed}', 'results', 'Full_KAN_Graphormer_result.pkl')
    r = pickle.load(open(p, 'rb'))
    er = r['eval_data']['evaluation_results']
    pred = np.array(er['predictions'])
    targ = np.array(er['targets'])
    return pred, targ


def pooled_r2(pred, targ, cols):
    p = pred[:, cols].ravel()
    t = targ[:, cols].ravel()
    return float(r2_score(t, p))


def main():
    idx = {c: i for i, c in enumerate(CITIES)}
    all_cols = list(range(len(CITIES)))
    t1_cols = [idx[c] for c in TIER1]
    t2_cols = [idx[c] for c in TIER2]
    remain_cols = [i for i in all_cols if i not in t1_cols and i not in t2_cols]

    groups = {'All': all_cols, 'Tier-1': t1_cols, 'Tier-2': t2_cols, 'Remaining12': remain_cols}
    res = {k: [] for k in groups}
    for s in range(N_SEEDS):
        pred, targ = load(s)
        for k, cols in groups.items():
            res[k].append(pooled_r2(pred, targ, cols))

    print('=== tab:spatial 分层 pooled R² (20 种子) ===')
    for k in ['All', 'Tier-1', 'Tier-2', 'Remaining12']:
        v = np.array(res[k])
        print(f'{k:14s}  R² = {v.mean():.4f} ± {v.std():.4f}')

    # 校验 seed_0 的 All 是否等于 0.7588 (确认 pooled 定义一致)
    pred, targ = load(0)
    s0_all = pooled_r2(pred, targ, all_cols)
    print(f'\n[校验] seed_0 All pooled R² = {s0_all:.4f}  (summary per_seed[0].r2 = 0.7588)')


if __name__ == '__main__':
    main()
