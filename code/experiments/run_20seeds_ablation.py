#!/usr/bin/env python
"""
20 随机种子 × 4 模型消融实验（无可视化，仅输出 R²/RMSE/MAE 的均值±标准差）

只跑 4 个模型（与导师要求一致）:
    Full_KAN_Graphormer, M_Graphormer_Baseline, KAN_Only, NoPhysics_KAN_Graphormer

输出:
    result/ablation_20seeds_4models_<时间戳>/
    ├── seed_<n>/models,results   # 每个种子的 checkpoint / 结果 pkl
    ├── summary_20seeds.csv        # 均值±标准差表格
    └── summary_20seeds.json       # 每个种子 × 每个模型的原始指标

用法:
    PYTHONIOENCODING=utf-8 python run_20seeds_ablation.py                 # 默认 20 种子 × 150 轮
    PYTHONIOENCODING=utf-8 python run_20seeds_ablation.py --seeds 1 --epochs 5   # 快速冒烟测试
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


MODEL_SUBSET = [
    'Full_KAN_Graphormer',
    'M_Graphormer_Baseline',
    'KAN_Only',
    'NoPhysics_KAN_Graphormer',
]
METRIC_KEYS = ['r2', 'rmse', 'mae']


def build_data():
    """加载原始数据（仅一次，内容与 run_ablation 完全一致，确定性无随机）"""
    features, adjs, pops, cities, dates = load_raw_data_only()
    lookback, predict = 7, 3
    total_samples = len(features) - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    return (features, adjs, pops, cities, dates,
            lookback, predict, total_samples, train_size, val_size)


def run_one_seed(seed, epochs, data, base_result_dir):
    features, adjs, pops, cities, dates, lookback, predict, total_samples, train_size, val_size = data
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import exp_lib.config as cfg
    seed_dir = os.path.join(base_result_dir, f"seed_{seed}")
    cfg.OUTPUT_DIR = seed_dir
    for sub in ["models", "results"]:
        os.makedirs(os.path.join(seed_dir, sub), exist_ok=True)

    # 训练/验证划分（依赖种子：shuffle）
    train_val_indices = list(range(train_size + val_size))
    np.random.shuffle(train_val_indices)
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    test_indices = list(range(train_size + val_size, total_samples))

    # 归一化器（基于训练集）
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

    # 接触矩阵（与 run_ablation 一致）
    raw_contact_matrix = torch.tensor([[10.0, 4.5, 0.8],
                                       [2.8, 11.5, 1.2],
                                       [0.6, 1.5, 3.5]], dtype=torch.float32).to(device)
    eigenvalues = torch.linalg.eigvals(raw_contact_matrix)
    spectral_radius = torch.max(torch.abs(eigenvalues)).real
    base_contact_matrix = (raw_contact_matrix / spectral_radius).float()
    pops = pops.float().to(device)

    ablation_config = {
        'num_nodes': len(cities),
        'input_dim': features.shape[-1],
        'device': device,
        'output_dir': seed_dir,
        'lookback': lookback,
        'hidden_dim': 64
    }

    manager = ImprovedAblationStudyManager(ablation_config)
    # 只保留 4 个目标模型
    all_defs = manager.model_definitions
    manager.model_definitions = {m: all_defs[m] for m in MODEL_SUBSET}

    results = manager.run_all_experiments(
        train_loader, val_loader, test_loader,
        base_contact_matrix, pops, cities, case_scaler, dates, epochs=epochs)

    metrics = {}
    for m in MODEL_SUBSET:
        pm = results[m]['eval_data']['performance_metrics']
        metrics[m] = {k: float(pm.get(k, float('nan'))) for k in METRIC_KEYS}
    return metrics


def main():
    parser = argparse.ArgumentParser(description='20种子×4模型消融（无可视化）')
    parser.add_argument('--seeds', type=int, default=20, help='种子数量 (默认 20)')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数 (默认 150)')
    parser.add_argument('--start', type=int, default=0, help='起始种子编号 (默认 0)')
    parser.add_argument('--base-dir', type=str, default=None, help='输出根目录 (默认 result/)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    if args.base_dir is None:
        args.base_dir = r"E:\Claude code\KAN+\result"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = os.path.join(args.base_dir, f"ablation_20seeds_4models_{timestamp}")
    os.makedirs(base_result_dir, exist_ok=True)
    print(f"输出根目录: {base_result_dir}")

    seeds = list(range(args.start, args.start + args.seeds))
    print(f"种子列表: {seeds}")
    print(f"模型: {MODEL_SUBSET}")

    data = build_data()

    all_metrics = {}
    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n{'#' * 80}\n### 种子 {i + 1}/{len(seeds)}  (seed={seed})\n{'#' * 80}")
        try:
            metrics = run_one_seed(seed, args.epochs, data, base_result_dir)
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

    # 汇总均值±标准差
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

    # 保存结果
    out_json = os.path.join(base_result_dir, 'summary_20seeds.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'seeds': [str(s) for s in seeds], 'epochs': args.epochs,
                   'per_seed': all_metrics, 'summary': summary}, f,
                  indent=2, ensure_ascii=False)
    pd.DataFrame(rows).to_csv(os.path.join(base_result_dir, 'summary_20seeds.csv'),
                              index=False, encoding='utf-8-sig')
    print(f"\n✅ 完成。有效种子 {len(valid_seeds)}/{len(seeds)}")
    print(f"   表格: {os.path.join(base_result_dir, 'summary_20seeds.csv')}")


if __name__ == '__main__':
    main()
