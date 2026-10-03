#!/usr/bin/env python
"""
20 种子 × 2 模型 容量对齐重跑 (NoPhysics_KAN_Graphormer + M_Graphormer_Baseline)。

目的: 消除消融中 hidden_dim(容量) 与 weight_decay(正则强度) 的混杂。
将 NoPhys 与 M-Graphormer 对齐到 Full KAN-G 的配置:
    hidden_dim = 32, weight_decay = 1e-2
其余模型 (Full / GAT / MLP / LSTM) 的结果沿用既有 20 种子实验, 不重跑。

数据切分逻辑与 run_20seeds_6models.py 完全一致 (set_seed + np.random.shuffle),
保证本轮重跑的 seed 切分与原实验一致, 结果可直接合并对比。

两种模式:
    --config align     (默认) 对齐容量: hidden_dim=32, weight_decay=1e-2
    --config original  校验模式: 用原配置 (hidden_dim=64, NoPhys 默认 wd=1e-4, M-Graph wd=1e-3)
                        用于验证本脚本能复现原实验的 seed 切分与结果

输出: result/ablation_20seeds_align_<时间戳>/summary_20seeds.json

用法:
    PYTHONIOENCODING=utf-8 python experiments/run_20seeds_align_capacity.py --seeds 1 --epochs 3 --config align      # 冒烟
    PYTHONIOENCODING=utf-8 python experiments/run_20seeds_align_capacity.py --seeds 1 --epochs 150 --config original  # 校验复现
    PYTHONIOENCODING=utf-8 python experiments/run_20seeds_align_capacity.py                                           # 完整 20 种子对齐重跑
"""
import sys
import os
import json
import argparse
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from exp_lib.config import set_seed
from exp_lib.data import load_raw_data_only, CityLevelLogMinMaxScaler, ImprovedTemporalDataset
from exp_lib.trainer import ImprovedAblationStudyManager
import exp_lib.config as cfg


MODEL_SUBSET = ['NoPhysics_KAN_Graphormer', 'M_Graphormer_Baseline']
METRIC_KEYS = ['r2', 'rmse', 'mae']

# 对齐目标 (与 Full KAN-G 一致)
ALIGN_HIDDEN_DIM = 32
ALIGN_WEIGHT_DECAY = 1e-2

# 原配置 (仅 --config original 校验用)
ORIG_HIDDEN_DIM = 64


def build_data():
    features, adjs, pops, cities, dates = load_raw_data_only()
    lookback, predict = 7, 3
    total_samples = len(features) - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    return (features, adjs, pops, cities, dates,
            lookback, predict, total_samples, train_size, val_size)


def run_one_seed(seed, epochs, data, base_result_dir, mode):
    features, adjs, pops, cities, dates, lookback, predict, total_samples, train_size, val_size = data
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_dir = os.path.join(base_result_dir, f"seed_{seed}")
    cfg.OUTPUT_DIR = seed_dir
    for sub in ["models", "results"]:
        os.makedirs(os.path.join(seed_dir, sub), exist_ok=True)

    # 每个种子只做一次切分 (与原实验一致)
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
                                       indices=train_indices, case_scaler=case_scaler,
                                       other_scaler=other_scaler)
    val_ds = ImprovedTemporalDataset(features, adjs, pops, lookback, predict,
                                     indices=val_indices, case_scaler=case_scaler,
                                     other_scaler=other_scaler)
    test_ds = ImprovedTemporalDataset(features, adjs, pops, lookback, predict,
                                      indices=test_indices, case_scaler=case_scaler,
                                      other_scaler=other_scaler)
    train_ds.training = True
    val_ds.training = False
    test_ds.training = False

    batch_size = min(8, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    raw_contact_matrix = torch.tensor([[10.0, 4.5, 0.8],
                                       [2.8, 11.5, 1.2],
                                       [0.6, 1.5, 3.5]], dtype=torch.float32).to(device)
    eigenvalues = torch.linalg.eigvals(raw_contact_matrix)
    spectral_radius = torch.max(torch.abs(eigenvalues)).real
    base_contact_matrix = (raw_contact_matrix / spectral_radius).float()
    pops = pops.float().to(device)

    hidden_dim = ALIGN_HIDDEN_DIM if mode == 'align' else ORIG_HIDDEN_DIM
    ablation_config = {
        'num_nodes': len(cities),
        'input_dim': features.shape[-1],
        'device': device,
        'output_dir': seed_dir,
        'lookback': lookback,
        'hidden_dim': hidden_dim,
    }

    manager = ImprovedAblationStudyManager(ablation_config)
    all_defs = manager.model_definitions
    manager.model_definitions = {m: all_defs[m] for m in MODEL_SUBSET}

    if mode == 'align':
        # 对齐到 Full 的容量与正则: hidden_dim=32, weight_decay=1e-2
        for m in MODEL_SUBSET:
            md = manager.model_definitions[m]
            md['args'] = dict(md['args'])
            md['args']['hidden_dim'] = ALIGN_HIDDEN_DIM
            md['weight_decay'] = ALIGN_WEIGHT_DECAY
    # original 模式: 不改任何东西, 走 trainer.py 默认 (NoPhys wd=1e-4, M-Graph wd=1e-3)

    results = manager.run_all_experiments(
        train_loader, val_loader, test_loader,
        base_contact_matrix, pops, cities, case_scaler, dates, epochs=epochs)

    metrics = {}
    for m in MODEL_SUBSET:
        pm = results[m]['eval_data']['performance_metrics']
        metrics[m] = {k: float(pm.get(k, float('nan'))) for k in METRIC_KEYS}
    return metrics


def main():
    parser = argparse.ArgumentParser(description='20种子×2模型 容量对齐重跑 (NoPhys + M-Graphormer)')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--config', type=str, default='align', choices=['align', 'original'])
    parser.add_argument('--base-dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  (cuda_available={torch.cuda.is_available()})")

    if args.base_dir is None:
        args.base_dir = r"E:\Claude code\KAN+\result"
    tag = 'align' if args.config == 'align' else 'original'
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = os.path.join(args.base_dir, f"ablation_20seeds_{tag}_{timestamp}")
    os.makedirs(base_result_dir, exist_ok=True)
    print(f"输出根目录: {base_result_dir}")
    print(f"模型: {MODEL_SUBSET}")
    if args.config == 'align':
        print(f"对齐配置: hidden_dim={ALIGN_HIDDEN_DIM}, weight_decay={ALIGN_WEIGHT_DECAY}")
    else:
        print("校验模式: 原配置 (hidden_dim=64, NoPhys wd=1e-4, M-Graph wd=1e-3)")

    seeds = list(range(args.start, args.start + args.seeds))
    data = build_data()

    all_metrics = {}
    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n{'#' * 80}\n### 种子 {i + 1}/{len(seeds)}  (seed={seed})\n{'#' * 80}")
        try:
            metrics = run_one_seed(seed, args.epochs, data, base_result_dir, args.config)
            all_metrics[str(seed)] = metrics
            dt = time.time() - t0
            print(f"  ✅ 种子 {seed} 完成，耗时 {dt:.1f}s ({dt / 60:.2f} min)")
            for m in MODEL_SUBSET:
                print(f"     {m:28s} R²={metrics[m]['r2']:.4f}  "
                      f"RMSE={metrics[m]['rmse']:.4f}  MAE={metrics[m]['mae']:.4f}")
        except Exception as e:
            print(f"  ❌ 种子 {seed} 出错: {e}")
            traceback.print_exc()
            all_metrics[str(seed)] = None

    print("\n" + "=" * 80)
    print("汇总：均值 ± 标准差（测试集）")
    print("=" * 80)
    valid_seeds = [s for s in all_metrics if all_metrics[s] is not None]
    summary = {}
    rows = []
    for m in MODEL_SUBSET:
        row = {'model': m}
        summary[m] = {}
        for k in METRIC_KEYS:
            vals = [all_metrics[s][m][k] for s in valid_seeds]
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1))
            else:
                mean = std = float('nan')
            row[f'{k}_mean'] = mean
            row[f'{k}_std'] = std
            summary[m][k] = {'mean': mean, 'std': std, 'n': len(vals)}
            print(f"  {m:28s} {k.upper():5s} = {mean:.4f} ± {std:.4f}  (n={len(vals)})")
        rows.append(row)

    out_json = os.path.join(base_result_dir, 'summary_20seeds.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'seeds': [str(s) for s in seeds], 'epochs': args.epochs,
                   'models': MODEL_SUBSET, 'mode': args.config,
                   'align_config': {'hidden_dim': ALIGN_HIDDEN_DIM, 'weight_decay': ALIGN_WEIGHT_DECAY},
                   'per_seed': all_metrics, 'summary': summary}, f,
                  indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(os.path.join(base_result_dir, 'summary_20seeds.csv'),
                              index=False, encoding='utf-8-sig')
    print(f"\n✅ 完成。有效种子 {len(valid_seeds)}/{len(seeds)}")
    print(f"   表格: {os.path.join(base_result_dir, 'summary_20seeds.csv')}")


if __name__ == '__main__':
    main()
