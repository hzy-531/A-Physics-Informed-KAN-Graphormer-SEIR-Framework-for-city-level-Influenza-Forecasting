#!/usr/bin/env python
"""Extract detailed metrics from ablation PKL files for journal-ready analysis"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Backward compat for old pickle files
import __main__
from exp_lib import models as _models
from exp_lib import layers as _layers
from exp_lib import physics as _physics
from exp_lib import loss as _loss
from exp_lib import data as _data

_classes_to_inject = [
    ('EnhancedFull_Graphormer_V3', _models.EnhancedFull_Graphormer_V3),
    ('EnhancedMLP_Baseline_V3', _models.EnhancedMLP_Baseline_V3),
    ('EnhancedLSTM_Baseline_V3', _models.EnhancedLSTM_Baseline_V3),
    ('EnhancedGAT_Baseline_V3', _models.EnhancedGAT_Baseline_V3),
    ('EnhancedKAN_Only_V3', _models.EnhancedKAN_Only_V3),
    ('M_Graphormer_Baseline', _models.M_Graphormer_Baseline),
    ('NoPhysics_KAN_Graphormer', _models.NoPhysics_KAN_Graphormer),
    ('RobustKANLinear', _layers.RobustKANLinear),
    ('StandardMLP', _layers.StandardMLP),
    ('GatedTCN', _layers.GatedTCN),
    ('GraphormerLayer', _layers.GraphormerLayer),
    ('ImprovedGATLayer', _layers.ImprovedGATLayer),
    ('compute_diffusion_distance', _layers.compute_diffusion_distance),
    ('RBFSpatialEncoding', _layers.RBFSpatialEncoding),
    ('age_structured_seir_step', _physics.age_structured_seir_step),
    ('CurriculumScientificLoss', _loss.CurriculumScientificLoss),
    ('CityLevelLogMinMaxScaler', _data.CityLevelLogMinMaxScaler),
    ('CityLevelLinearMinMaxScaler', _data.CityLevelLinearMinMaxScaler),
    ('TemporalFeatureEnhancer', _data.TemporalFeatureEnhancer),
    ('ImprovedTemporalDataset', _data.ImprovedTemporalDataset),
    ('DataCollector', _data.DataCollector),
    ('load_raw_data_only', _data.load_raw_data_only),
    ('ImprovedActualDataVisualizer', __import__('exp_lib.visualizer', fromlist=['ImprovedActualDataVisualizer']).ImprovedActualDataVisualizer),
    ('ImprovedAblationStudyManager', __import__('exp_lib.trainer', fromlist=['ImprovedAblationStudyManager']).ImprovedAblationStudyManager),
]
for name, cls in _classes_to_inject:
    setattr(__main__, name, cls)

import pickle
import numpy as np
import json

RESULT_DIR = r'E:\Claude code\KAN+\result\ablation_study_v9_visual_20260617_193138\results'
MODELS = ['MLP_Baseline', 'LSTM_Baseline', 'GAT_Baseline', 'M_Graphormer_Baseline',
          'KAN_Only', 'NoPhysics_KAN_Graphormer', 'Full_KAN_Graphormer']
CITIES = ['北京市','天津市','上海市','重庆市','广州市','深圳市','西安市','成都市',
          '武汉市','杭州市','南京市','苏州市','无锡市','郑州市','长沙市','沈阳市',
          '大连市','青岛市','济南市','宁波市','厦门市','哈尔滨市','长春市','石家庄市']

def extract_all():
    all_data = {}
    for model_name in MODELS:
        pkl_path = os.path.join(RESULT_DIR, f'{model_name}_result.pkl')
        with open(pkl_path, 'rb') as f:
            try:
                data = pickle.load(f)
            except AttributeError as e:
                # Stub missing class and retry
                missing = str(e).split("'")[1] if "'" in str(e) else str(e).split()[-1]
                print(f'  Stubbing missing class: {missing}')
                setattr(__main__, missing, type(missing, (), {}))
                f.seek(0)
                data = pickle.load(f)

        info = {}
        eval_data = data.get('eval_data', {})
        metrics = eval_data.get('performance_metrics', {})
        info['r2'] = float(metrics.get('r2', 0))
        info['rmse'] = float(metrics.get('rmse', 0))
        info['mae'] = float(metrics.get('mae', 0))

        per_city = metrics.get('per_city_r2', {})
        if per_city:
            info['per_city_r2'] = {str(k): float(v) for k, v in per_city.items()}

        pv = eval_data.get('param_validity', {})
        if pv:
            info['physics'] = {
                'beta_mean': float(pv.get('beta_mean', 0)),
                'beta_std': float(pv.get('beta_std', 0)),
                'gamma_mean': float(pv.get('gamma_mean', 0)),
                'gamma_std': float(pv.get('gamma_std', 0)),
                'r0_mean': float(pv.get('r0_mean', 0)),
                'r0_std': float(pv.get('r0_std', 0)),
                'r0_valid_ratio': float(pv.get('r0_valid_ratio', 0)),
            }

        th = data.get('train_history', {})
        if th:
            train_loss = th.get('train_loss', [])
            val_loss = th.get('val_loss', [])
            if train_loss:
                info['train_loss_final'] = float(train_loss[-1])
                info['train_loss_initial'] = float(train_loss[0])
            if val_loss:
                info['val_loss_min'] = float(min(val_loss))
                info['val_loss_final'] = float(val_loss[-1])
                info['val_overfit_gap'] = float(val_loss[-1] - min(val_loss))

        fp = data.get('full_pred', None)
        if fp:
            for key in ['beta_full', 'contact_full', 'rt_full']:
                arr = fp.get(key, None)
                if arr is not None:
                    if hasattr(arr, 'detach'):
                        arr = arr.detach().cpu().numpy()
                    elif not isinstance(arr, np.ndarray):
                        arr = np.array(arr)
                    info[f'{key}_mean'] = float(arr.mean())
                    info[f'{key}_std'] = float(arr.std())
                    info[f'{key}_max'] = float(arr.max())

        all_data[model_name] = info
        print(f'Loaded {model_name}: R²={info["r2"]:.4f}')

    # Save as JSON
    out_path = os.path.join(os.path.dirname(RESULT_DIR), 'extracted_metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f'\nSaved to {out_path}')
    return all_data

if __name__ == '__main__':
    extract_all()
