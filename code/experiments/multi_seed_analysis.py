#!/usr/bin/env python
"""
从多个种子的消融实验结果计算均值±标准差，生成带误差条的论文表格
用法: PYTHONIOENCODING=utf-8 python multi_seed_analysis.py
"""
import pickle, os, json, sys
import numpy as np

# 收集所有种子结果
SEED_DIRS = [
    'E:/Claude code/KAN+/result/ablation_study_v9_20260630_195959',  # seed 777
    # 添加更多种子目录...
]

def load_all_seeds(dirs):
    """加载所有种子的结果，按模型名组织"""
    all_results = {}  # {model_name: [r2_list, rmse_list, ...]}

    for d in dirs:
        results_path = os.path.join(d, 'results')
        if not os.path.exists(results_path):
            print(f'  SKIP: {results_path} not found')
            continue

        seed_name = os.path.basename(d)
        print(f'Loading {seed_name}...')

        for f in sorted(os.listdir(results_path)):
            if not f.endswith('_result.pkl'):
                continue
            model = f.replace('_result.pkl', '')
            with open(os.path.join(results_path, f), 'rb') as fh:
                data = pickle.load(fh)

            perf = data.get('eval_data', {}).get('performance_metrics', {})
            err = data.get('eval_data', {}).get('error_analysis', {})
            mi = data.get('model_info', {})

            if model not in all_results:
                all_results[model] = {
                    'r2': [], 'rmse': [], 'mae': [],
                    'r2_t1': [], 'r2_other': [],
                    'dw': [], 'skew': [],
                    'resid_mean': [], 'resid_std': [],
                    'params': mi.get('trainable_params', 0),
                }

            all_results[model]['r2'].append(float(perf.get('r2', 0)))
            all_results[model]['rmse'].append(float(perf.get('rmse', 0)))
            all_results[model]['mae'].append(float(perf.get('mae', 0)))
            all_results[model]['r2_t1'].append(float(perf.get('r2_tier1', 0)))
            all_results[model]['r2_other'].append(float(perf.get('r2_tier2', 0)))
            all_results[model]['dw'].append(float(err.get('durbin_watson', 0)))
            all_results[model]['skew'].append(float(err.get('residual_skewness', 0)))
            all_results[model]['resid_mean'].append(float(err.get('residual_mean', 0)))
            all_results[model]['resid_std'].append(float(err.get('residual_std', 0)))

    return all_results


def compute_stats(all_results):
    """计算均值±标准差"""
    stats = {}
    for model, metrics in all_results.items():
        n = len(metrics['r2'])
        if n < 1:
            continue
        stats[model] = {
            'r2_mean': np.mean(metrics['r2']),
            'r2_std': np.std(metrics['r2']),
            'rmse_mean': np.mean(metrics['rmse']),
            'rmse_std': np.std(metrics['rmse']),
            'mae_mean': np.mean(metrics['mae']),
            'mae_std': np.std(metrics['mae']),
            'r2_t1_mean': np.mean(metrics['r2_t1']),
            'r2_t1_std': np.std(metrics['r2_t1']),
            'r2_other_mean': np.mean(metrics['r2_other']),
            'r2_other_std': np.std(metrics['r2_other']),
            'dw_mean': np.mean(metrics['dw']),
            'dw_std': np.std(metrics['dw']),
            'skew_mean': np.mean(metrics['skew']),
            'skew_std': np.std(metrics['skew']),
            'n_seeds': n,
            'params': metrics['params'],
        }
    return stats


def print_table(stats):
    """打印带误差条的排名表"""
    model_names = {
        'Full_KAN_Graphormer': 'Full KAN-G',
        'GAT_Baseline': 'GAT',
        'MLP_Baseline': 'MLP',
        'M_Graphormer_Baseline': 'M-Graphormer',
        'KAN_Only': 'KAN-Only',
        'NoPhysics_KAN_Graphormer': 'NoPhys-KAN-G',
        'LSTM_Baseline': 'LSTM',
    }

    ranked = sorted(stats.items(), key=lambda x: x[1]['r2_mean'], reverse=True)

    print()
    print('=' * 110)
    print(f'MULTI-SEED ABLATION RESULTS ({stats[list(stats.keys())[0]]["n_seeds"]} seeds)')
    print('=' * 110)
    print(f'{"Rank":<5} {"Model":<18} {"R²":<18} {"RMSE":<18} {"MAE":<18} {"DW":<14} {"Skew":<14}')
    print('-' * 110)

    for rank, (model, s) in enumerate(ranked, 1):
        name = model_names.get(model, model)
        marker = ' ★' if rank == 1 else ''
        print(f'{rank:<5} {name:<18} {s["r2_mean"]:.4f}±{s["r2_std"]:.4f}    '
              f'{s["rmse_mean"]:.3f}±{s["rmse_std"]:.3f}    '
              f'{s["mae_mean"]:.3f}±{s["mae_std"]:.3f}    '
              f'{s["dw_mean"]:.3f}±{s["dw_std"]:.3f}   '
              f'{s["skew_mean"]:.3f}±{s["skew_std"]:.3f}{marker}')

    # 边际效应
    full = stats.get('Full_KAN_Graphormer', {})
    mg = stats.get('M_Graphormer_Baseline', {})
    npkg = stats.get('NoPhysics_KAN_Graphormer', {})
    ko = stats.get('KAN_Only', {})

    if all([full, mg, npkg, ko]):
        d_kan = full['r2_mean'] - mg['r2_mean']
        d_phys = full['r2_mean'] - npkg['r2_mean']
        d_graph = full['r2_mean'] - ko['r2_mean']

        # 传播误差
        d_kan_std = np.sqrt(full['r2_std']**2 + mg['r2_std']**2)
        d_phys_std = np.sqrt(full['r2_std']**2 + npkg['r2_std']**2)
        d_graph_std = np.sqrt(full['r2_std']**2 + ko['r2_std']**2)

        print()
        print('MARGINAL EFFECTS (mean ± propagated std):')
        print(f'  Δ_KAN   = {d_kan:+.4f} ± {d_kan_std:.4f}')
        print(f'  Δ_Phys  = {d_phys:+.4f} ± {d_phys_std:.4f}')
        print(f'  Δ_Graph = {d_graph:+.4f} ± {d_graph_std:.4f}')

    return stats


if __name__ == '__main__':
    # 自动发现最新结果目录
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        # 从命令行参数或自动发现
        result_base = 'E:/Claude code/KAN+/result'
        candidate_dirs = []
        for d in sorted(os.listdir(result_base)):
            full_path = os.path.join(result_base, d)
            results_path = os.path.join(full_path, 'results')
            if os.path.isdir(full_path) and os.path.exists(results_path):
                # 检查是否有完整的结果 (>=7 pkl files)
                pkl_count = len([f for f in os.listdir(results_path) if f.endswith('_result.pkl')])
                if pkl_count >= 7:
                    candidate_dirs.append(full_path)

        print(f'Found {len(candidate_dirs)} result directories with complete results.')
        print('Please specify directories manually or use the latest ones.')

        if candidate_dirs:
            # 使用最近的三个
            dirs = candidate_dirs[-3:]
            print(f'Using: {dirs}')

    if not dirs:
        print('No result directories found!')
        sys.exit(1)

    all_results = load_all_seeds(dirs)
    stats = compute_stats(all_results)
    print_table(stats)

    # 保存
    output_path = os.path.join(os.path.dirname(dirs[0]), 'multi_seed_stats.json')
    # Convert numpy types for JSON serialization
    json_stats = {}
    for model, s in stats.items():
        json_stats[model] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in s.items()}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)
    print(f'\n✅ Stats saved to {output_path}')
