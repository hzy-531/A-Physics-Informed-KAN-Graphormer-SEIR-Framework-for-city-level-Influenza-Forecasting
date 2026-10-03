#!/usr/bin/env python
"""
极简调试脚本 — 只训练 1 个模型 × 1 种子
改模型代码后快速验证，不用等 7 个模型跑完

用法:
    PYTHONIOENCODING=utf-8 python run_single.py                          # 默认 Full_KAN_Graphormer
    PYTHONIOENCODING=utf-8 python run_single.py --model KAN_Only          # 指定模型
    PYTHONIOENCODING=utf-8 python run_single.py --epochs 30 --seed 42    # 30轮+指定种子
    PYTHONIOENCODING=utf-8 python run_single.py --no-plot                 # 只训练不画图
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from exp_lib.config import (set_seed, create_output_dir, TARGET_CITIES,
                            TIER_1_CITIES, TIER_2_CITIES, LOCKDOWN_INFO,
                            FILE_CITY_FEAT)
from exp_lib.data import (load_raw_data_only, CityLevelLogMinMaxScaler,
                          ImprovedTemporalDataset, add_city_specific_features)
from exp_lib.trainer import ImprovedAblationStudyManager
from exp_lib.visualizer import ImprovedActualDataVisualizer


def main(model_name='Full_KAN_Graphormer', seed=5780, epochs=None,
         output_dir=None, skip_plot=False):
    t0 = time.time()

    # 1. 种子 + 输出目录
    set_seed(seed)
    if output_dir is None:
        output_dir = create_output_dir()
    import exp_lib.config as cfg
    cfg.OUTPUT_DIR = output_dir

    print(f"模型: {model_name}  种子: {seed}  输出: {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 2. 数据
    features, adjs, pops, cities, dates = load_raw_data_only()
    lookback, predict = 7, 3
    total_samples = len(features) - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)

    train_val_indices = list(range(train_size + val_size))
    np.random.shuffle(train_val_indices)
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    test_indices = list(range(train_size + val_size, total_samples))

    # 归一化器
    train_days_set = set()
    for idx in train_indices:
        for t in range(idx, idx + lookback + predict):
            train_days_set.add(t)
    train_days = sorted(train_days_set)
    train_features = features[train_days]

    case_scaler = CityLevelLogMinMaxScaler()
    case_scaler.fit(train_features[:, :, :3].numpy() if isinstance(train_features, torch.Tensor)
                    else train_features[:, :, :3])
    other_scaler = CityLevelLogMinMaxScaler()
    other_scaler.fit(train_features[:, :, 3:].numpy() if isinstance(train_features, torch.Tensor)
                     else train_features[:, :, 3:])

    # 数据集
    train_ds = ImprovedTemporalDataset(features, adjs, pops, lookback, predict,
                                       indices=train_indices,
                                       case_scaler=case_scaler, other_scaler=other_scaler)
    val_ds = ImprovedTemporalDataset(features, adjs, pops, lookback, predict,
                                     indices=val_indices,
                                     case_scaler=case_scaler, other_scaler=other_scaler)
    test_ds = ImprovedTemporalDataset(features, adjs, pops, lookback, predict,
                                      indices=test_indices,
                                      case_scaler=case_scaler, other_scaler=other_scaler)
    train_ds.training = True
    val_ds.training = False
    test_ds.training = False
    batch_size = min(8, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 接触矩阵
    raw_contact = torch.tensor([[10.0, 4.5, 0.8], [2.8, 11.5, 1.2], [0.6, 1.5, 3.5]],
                               dtype=torch.float32).to(device)
    eigenvalues = torch.linalg.eigvals(raw_contact)
    spectral_radius = torch.max(torch.abs(eigenvalues)).real
    base_contact_matrix = (raw_contact / spectral_radius).float()
    pops = pops.float().to(device)

    # 3. 配置 — 只保留目标模型
    ablation_config = {
        'num_nodes': len(cities),
        'input_dim': features.shape[-1],
        'device': device,
        'output_dir': output_dir,
        'lookback': lookback,
        'hidden_dim': 64
    }
    manager = ImprovedAblationStudyManager(ablation_config)

    # 过滤：只保留目标模型
    all_defs = manager.model_definitions
    if model_name not in all_defs:
        print(f"未知模型 '{model_name}'，可选: {list(all_defs.keys())}")
        return
    manager.model_definitions = {model_name: all_defs[model_name]}
    print(f"已过滤，只保留: {model_name}")

    # 4. 用 run_all_experiments 一站式跑完 (setup + train + eval + full_pred)
    #    因为 model_definitions 已过滤，只会跑 1 个模型
    results = manager.run_all_experiments(
        train_loader, val_loader, test_loader,
        base_contact_matrix, pops, cities,
        case_scaler, dates,
        epochs=epochs
    )

    eval_data = results[model_name]['eval_data']
    train_history = results[model_name]['train_history']
    full_pred = results[model_name].get('full_pred', None)
    metrics = eval_data.get('performance_metrics', {})

    print(f"\n{'='*50}")
    print(f"  {model_name}  种子={seed}")
    print(f"  R²={metrics.get('r2', 0):.4f}  RMSE={metrics.get('rmse', 0):.4f}  "
          f"MAE={metrics.get('mae', 0):.4f}")
    print(f"{'='*50}")

    # 5. 画图
    if not skip_plot:
        print("\n>>> 生成图表...")
        viz = ImprovedActualDataVisualizer(output_dir, cities)

        train_data = {
            'training_history': train_history,
            'model_parameters': eval_data.get('model_parameters', {})
        }
        viz.plot_actual_training_history(train_data, model_name)
        viz.plot_actual_prediction_results(eval_data, model_name)
        viz.plot_error_diagnostics(eval_data, model_name)

        if manager.models[model_name]['has_physics']:
            viz.plot_actual_parameter_analysis(eval_data, model_name)

        # 全时段参数对比图 (武汉 vs 青岛，左右并排)
        print("\n>>> 封城参数对比图...")
        viz._plot_lockdown_params_comparison(results)

    elapsed = time.time() - t0
    print(f"\n✅ 完成! 耗时 {elapsed:.0f}s  输出: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='极简调试 — 单模型×单种子')
    parser.add_argument('--model', type=str, default='Full_KAN_Graphormer',
                        help='模型名 (默认: Full_KAN_Graphormer)')
    parser.add_argument('--seed', type=int, default=5780)
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认: 模型预设值)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-plot', action='store_true', help='跳过画图')
    args = parser.parse_args()

    main(model_name=args.model, seed=args.seed, epochs=args.epochs,
         output_dir=args.output, skip_plot=args.no_plot)
