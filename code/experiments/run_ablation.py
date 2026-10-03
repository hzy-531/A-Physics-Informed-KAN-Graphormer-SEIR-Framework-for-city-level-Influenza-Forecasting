#!/usr/bin/env python
"""
单次消融实验 — 7 模型 × 1 种子
用法:
    PYTHONIOENCODING=utf-8 python run_ablation.py              # 默认种子 5780
    PYTHONIOENCODING=utf-8 python run_ablation.py --seed 368   # 指定种子
    PYTHONIOENCODING=utf-8 python run_ablation.py --epochs 50  # 快速测试(50轮)
"""
import sys
import os
import argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from exp_lib.config import set_seed, create_output_dir, TARGET_CITIES, TIER_1_CITIES, \
    TIER_2_CITIES, LOCKDOWN_INFO, FILE_CITY_FEAT, OUTPUT_DIR as GLOBAL_OUTPUT_DIR
from exp_lib.data import (load_raw_data_only, CityLevelLogMinMaxScaler,
                          ImprovedTemporalDataset, add_city_specific_features)
from exp_lib.trainer import ImprovedAblationStudyManager
from exp_lib.visualizer import ImprovedActualDataVisualizer


def main(seed=5780, epochs=None, output_dir=None):
    # 1. 设置随机种子
    set_seed(seed)

    # 2. 创建输出目录
    if output_dir is None:
        output_dir = create_output_dir()
    # 更新全局 OUTPUT_DIR (trainer 内部通过 _config.OUTPUT_DIR 引用)
    import exp_lib.config as cfg
    cfg.OUTPUT_DIR = output_dir

    print(f"输出目录: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 加载原始数据
    features, adjs, pops, cities, dates = load_raw_data_only()
    print(f"原始特征形状: {features.shape}")
    print(f"邻接矩阵形状: {adjs.shape}")
    print(f"城市数量: {len(cities)}")

    # 4. 时序划分
    lookback, predict = 7, 3
    total_samples = len(features) - lookback - predict + 1
    train_size = int(total_samples * 0.6)
    val_size = int(total_samples * 0.2)
    test_size = total_samples - train_size - val_size

    train_val_indices = list(range(train_size + val_size))
    np.random.shuffle(train_val_indices)
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    test_indices = list(range(train_size + val_size, total_samples))

    # 5. 基于训练集拟合归一化器
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

    # 6. 创建数据集
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

    # 7. 接触矩阵
    raw_contact_matrix = torch.tensor([[10.0, 4.5, 0.8],
                                       [2.8, 11.5, 1.2],
                                       [0.6, 1.5, 3.5]], dtype=torch.float32).to(device)
    eigenvalues = torch.linalg.eigvals(raw_contact_matrix)
    spectral_radius = torch.max(torch.abs(eigenvalues)).real
    base_contact_matrix = (raw_contact_matrix / spectral_radius).float()
    pops = pops.float().to(device)

    # 8. 消融实验配置
    ablation_config = {
        'num_nodes': len(cities),
        'input_dim': features.shape[-1],
        'device': device,
        'output_dir': output_dir,
        'lookback': lookback,
        'hidden_dim': 64
    }

    # 9. 运行 7 模型消融
    ablation_manager = ImprovedAblationStudyManager(ablation_config)
    results = ablation_manager.run_all_experiments(
        train_loader, val_loader, test_loader,
        base_contact_matrix, pops, cities,
        case_scaler, dates,
        epochs=epochs
    )

    # 10. 可视化
    viz_manager = ImprovedActualDataVisualizer(output_dir, cities)

    # 构建封城索引
    lockdown_idx = {}
    for city, info in LOCKDOWN_INFO.items():
        if city in cities and info['start'] is not None and info['end'] is not None:
            try:
                start_idx = dates.index(pd.Timestamp(info['start']))
                end_idx = dates.index(pd.Timestamp(info['end']))
                lockdown_idx[city] = {'start_idx': start_idx, 'end_idx': end_idx}
            except:
                pass

    # 构建城市特征 DataFrame
    try:
        df_static = pd.read_excel(FILE_CITY_FEAT)
    except:
        df_static = pd.DataFrame({
            '城市名称': cities,
            '2019市常住人口(万人)': [1000] * len(cities),
            '区域标识 (Region)(南方=1，北方=0)': [1] * len(cities)
        })
    city_features_df = pd.DataFrame({
        '城市': cities,
        '人口密度': [add_city_specific_features(df_static, c)['population_density'] / 5000.0
                     for c in cities],
        '城市等级': [1 if c in TIER_1_CITIES else 2 if c in TIER_2_CITIES else 3
                     for c in cities],
        '供暖地区': [1 if c in ['北京市', '天津市', '石家庄市', '沈阳市', '哈尔滨市', '长春市']
                     else 0 for c in cities],
        '封城强度': [LOCKDOWN_INFO.get(c, {}).get('strength', 0.0) for c in cities]
    })

    # 遍历每个模型生成图表
    for model_name, result in results.items():
        print(f"    正在为 {model_name} 生成图表...")
        train_data = {
            'training_history': result.get('train_history', {}),
            'model_parameters': result.get('collector_data', {}).get('model_parameters', {})
        }
        eval_data = result.get('eval_data', {})

        viz_manager.plot_actual_model_architecture(train_data, model_name)
        viz_manager.plot_actual_training_history(train_data, model_name)
        viz_manager.plot_actual_prediction_results(eval_data, model_name)
        viz_manager.plot_error_diagnostics(eval_data, model_name)
        viz_manager.plot_kan_splines(result['model_info']['model'], model_name)

        if result['model_info']['has_physics']:
            viz_manager.plot_actual_parameter_analysis(eval_data, model_name)
            viz_manager.plot_parameter_evolution(eval_data, model_name, dates, lockdown_idx)
            viz_manager.plot_epidemiological_insights(eval_data, model_name, LOCKDOWN_INFO)

        if 'Graphormer' in model_name or 'GAT' in model_name:
            viz_manager.plot_actual_attention_weights(eval_data, model_name)

        viz_manager.print_quantitative_insights(eval_data, city_features_df, model_name)
        viz_manager.generate_academic_thesis_report(eval_data, model_name, city_features_df)

    # 消融实验总对比图
    print("\n>>> [可视化] 生成消融实验对比图...")
    viz_manager.plot_ablation_comparison(results)

    # 11. 保存结果
    save_results(results, ablation_config, cities, dates, total_samples,
                 train_size, val_size, test_size, output_dir)

    # 12. 全时段参数对比图 (封城 vs 未封城，左右并排)
    print("\n>>> 正在绘制封城参数对比图...")
    viz_manager._plot_lockdown_params_comparison(results)

    print(f"\n✅ 单次消融实验完成! 结果保存在: {output_dir}")
    return results


def save_results(results, config, cities, dates, total_samples, train_size, val_size,
                 test_size, output_dir):
    import json
    config_serializable = {}
    for k, v in config.items():
        if isinstance(v, torch.device):
            config_serializable[k] = str(v)
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            config_serializable[k] = v.tolist()
        else:
            config_serializable[k] = v

    summary = {
        'config': config_serializable,
        'cities': cities,
        'dates': [str(d) for d in dates],
        'total_samples': total_samples,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'results': {}
    }
    for model_name, res in results.items():
        metrics = res.get('eval_data', {}).get('performance_metrics', {})
        error_analysis = res.get('eval_data', {}).get('error_analysis', {})
        summary['results'][model_name] = {
            'test_r2': float(metrics.get('r2', 0)),
            'test_rmse': float(metrics.get('rmse', 0)),
            'test_mae': float(metrics.get('mae', 0)),
            'param_validity': res.get('eval_data', {}).get('param_validity', {}),
            'error_analysis': {
                'residual_mean': error_analysis.get('residual_mean', 0),
                'residual_std': error_analysis.get('residual_std', 0),
                'residual_skewness': error_analysis.get('residual_skewness', 0),
                'durbin_watson': error_analysis.get('durbin_watson', 0),
                'per_city_errors': error_analysis.get('per_city_errors', {})
            }
        }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"结果已保存至 {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='单次消融实验 — 7模型×1种子')
    parser.add_argument('--seed', type=int, default=5780, help='随机种子 (默认: 5780)')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数 (默认: 各模型预设值)')
    parser.add_argument('--output', type=str, default=None, help='输出目录 (默认: 自动创建)')
    args = parser.parse_args()

    main(seed=args.seed, epochs=args.epochs, output_dir=args.output)
