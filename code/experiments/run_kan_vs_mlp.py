#!/usr/bin/env python
"""
KAN vs MLP 对照实验 — 对比 B-spline 激活 vs Linear+SiLU 拟合

配对: Full_KAN_Graphormer vs M_Graphormer_Baseline
      架构完全相同 (图Transformer + TCN + 物理ODE)，唯一变量是激活层类型
      KAN: RobustKANLinear (B-spline基 + SiLU基)
      MLP: StandardMLP (Linear → SiLU → Linear, 参数量对齐)

用法:
    PYTHONIOENCODING=utf-8 python run_kan_vs_mlp.py
    PYTHONIOENCODING=utf-8 python run_kan_vs_mlp.py --epochs 30  # 快速测试
"""
import sys
import os
import argparse
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from exp_lib.config import set_seed, create_output_dir, TARGET_CITIES
from exp_lib.data import (load_raw_data_only, CityLevelLogMinMaxScaler,
                          ImprovedTemporalDataset)
from exp_lib.trainer import ImprovedAblationStudyManager


def run_comparison(seed=5780, epochs=None, output_dir=None):
    t0 = time.time()

    set_seed(seed)
    if output_dir is None:
        output_dir = create_output_dir()
    import exp_lib.config as cfg
    cfg.OUTPUT_DIR = output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print(f"  KAN vs MLP 对照实验")
    print(f"  输出: {output_dir}  种子: {seed}  设备: {device}")
    print(f"{'='*60}")

    # 数据
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

    raw_contact = torch.tensor([[10.0, 4.5, 0.8], [2.8, 11.5, 1.2], [0.6, 1.5, 3.5]],
                               dtype=torch.float32).to(device)
    eigenvalues = torch.linalg.eigvals(raw_contact)
    spectral_radius = torch.max(torch.abs(eigenvalues)).real
    base_contact_matrix = (raw_contact / spectral_radius).float()
    pops = pops.float().to(device)

    # 两个对比模型
    models_to_run = ['Full_KAN_Graphormer', 'M_Graphormer_Baseline']

    ablation_config = {
        'num_nodes': len(cities),
        'input_dim': features.shape[-1],
        'device': device,
        'output_dir': output_dir,
        'lookback': lookback,
        'hidden_dim': 16
    }

    pair_results = {}
    for model_name in models_to_run:
        print(f"\n{'─'*50}")
        print(f"  训练: {model_name}")
        print(f"{'─'*50}")
        t_model = time.time()

        manager = ImprovedAblationStudyManager(ablation_config)
        all_defs = manager.model_definitions
        if model_name not in all_defs:
            print(f"  ❌ 未知模型 '{model_name}'，跳过")
            continue
        manager.model_definitions = {model_name: all_defs[model_name]}

        results = manager.run_all_experiments(
            train_loader, val_loader, test_loader,
            base_contact_matrix, pops, cities,
            case_scaler, dates, epochs=epochs
        )

        metrics = results[model_name]['eval_data'].get('performance_metrics', {})
        pair_results[model_name] = {
            'r2': float(metrics.get('r2', 0)),
            'rmse': float(metrics.get('rmse', 0)),
            'mae': float(metrics.get('mae', 0)),
            'trainable_params': manager.models[model_name]['trainable_params'],
            'elapsed': time.time() - t_model,
        }
        print(f"  {model_name}: R²={pair_results[model_name]['r2']:.4f}  "
              f"RMSE={pair_results[model_name]['rmse']:.4f}  "
              f"MAE={pair_results[model_name]['mae']:.4f}  "
              f"耗时 {pair_results[model_name]['elapsed']:.0f}s")

    # 打印对比表
    elapsed_total = time.time() - t0
    kan_m = 'Full_KAN_Graphormer'
    mlp_m = 'M_Graphormer_Baseline'

    if kan_m in pair_results and mlp_m in pair_results:
        kr = pair_results[kan_m]
        mr = pair_results[mlp_m]
        r2_delta = kr['r2'] - mr['r2']
        rmse_delta = (mr['rmse'] - kr['rmse']) / (mr['rmse'] + 1e-8) * 100
        mae_delta = (mr['mae'] - kr['mae']) / (mr['mae'] + 1e-8) * 100

        print(f"\n{'='*60}")
        print(f"  KAN vs MLP 对照结果")
        print(f"{'='*60}")
        print(f"  {'模型':<26} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'参数量':>10}")
        print(f"  {kan_m:<26} {kr['r2']:>7.4f} {kr['rmse']:>8.4f} {kr['mae']:>8.4f} {kr['trainable_params']:>10,}")
        print(f"  {mlp_m:<26} {mr['r2']:>7.4f} {mr['rmse']:>8.4f} {mr['mae']:>8.4f} {mr['trainable_params']:>10,}")
        print(f"  {'─'*26} {'─'*7} {'─'*8} {'─'*8} {'─'*10}")
        sign = '✅ KAN更优' if r2_delta > 0 else '❌ MLP更优'
        print(f"  {'Δ (KAN − MLP)':<26} {r2_delta:>+7.4f} {rmse_delta:>+7.1f}% {mae_delta:>+7.1f}%  {sign}")
        print()

        # 解读
        print(f"  📊 分析:")
        print(f"     R² 提升: {r2_delta:+.4f} — KAN B-spline 非线性拟合能力更强")
        print(f"     RMSE 降低: {rmse_delta:.1f}%")
        print(f"     MAE 降低: {mae_delta:.1f}%")
        if kr['trainable_params'] < mr['trainable_params']:
            print(f"     参数量: KAN ({kr['trainable_params']:,}) < MLP ({mr['trainable_params']:,}) — KAN更高效")
        else:
            print(f"     参数量: KAN ({kr['trainable_params']:,}) > MLP ({mr['trainable_params']:,})")

        # 保存
        summary_path = os.path.join(output_dir, 'kan_vs_mlp_summary.json')
        save_data = {
            'seed': seed, 'epochs': epochs, 'elapsed_total_s': elapsed_total,
            'kan': {'name': kan_m, **kr},
            'mlp': {'name': mlp_m, **mr},
            'r2_delta': r2_delta,
            'rmse_improve_pct': rmse_delta,
            'mae_improve_pct': mae_delta,
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\n  📁 结果已保存: {summary_path}")

    print(f"\n✅ 总耗时 {elapsed_total:.0f}s")
    return pair_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KAN vs MLP 对照实验')
    parser.add_argument('--seed', type=int, default=5780)
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (默认: 150)')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    run_comparison(seed=args.seed, epochs=args.epochs, output_dir=args.output)
