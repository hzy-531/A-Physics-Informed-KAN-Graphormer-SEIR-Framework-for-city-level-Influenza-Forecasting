#!/usr/bin/env python
"""
独立画图工具 — 读取已有 pkl 结果，生成全部图表，无需重新训练

用法:
    # 从单个结果目录生成图表
    PYTHONIOENCODING=utf-8 python plot_results.py --dir <result_dir>

    # 只生成消融对比图 (需要 result 目录下有各模型的 pkl 文件)
    PYTHONIOENCODING=utf-8 python plot_results.py --dir <result_dir> --comparison-only

    # 为单个模型生成图表
    PYTHONIOENCODING=utf-8 python plot_results.py --dir <result_dir> --model Full_KAN_Graphormer

    # 从多个种子目录生成汇总对比
    PYTHONIOENCODING=utf-8 python plot_results.py --dirs dir1 dir2 dir3 --summary
"""
import sys
import os
import argparse
import pickle
import json
import glob
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_lib.config import (TARGET_CITIES, TIER_1_CITIES, TIER_2_CITIES,
                            LOCKDOWN_INFO, FILE_CITY_FEAT)
from exp_lib.data import add_city_specific_features
from exp_lib.visualizer import ImprovedActualDataVisualizer


# ============================================================
# 向后兼容: 旧 pkl 文件中的类定义在 __main__ 模块中，
# 需要注入到 __main__ 以便 pickle 能正确反序列化
# ============================================================
def _setup_backward_compat():
    """将模型类注册到 __main__ 以兼容旧 pkl 文件"""
    import __main__
    from exp_lib import models as _models
    from exp_lib import layers as _layers
    from exp_lib import physics as _physics
    from exp_lib import loss as _loss
    from exp_lib import data as _data

    # 所有可能在旧 pkl 中出现的类
    _classes = [
        # 模型
        ('EnhancedFull_Graphormer_V3', _models.EnhancedFull_Graphormer_V3),
        ('EnhancedMLP_Baseline_V3', _models.EnhancedMLP_Baseline_V3),
        ('EnhancedLSTM_Baseline_V3', _models.EnhancedLSTM_Baseline_V3),
        ('EnhancedGAT_Baseline_V3', _models.EnhancedGAT_Baseline_V3),
        ('EnhancedKAN_Only_V3', _models.EnhancedKAN_Only_V3),
        ('M_Graphormer_Baseline', _models.M_Graphormer_Baseline),
        ('NoPhysics_KAN_Graphormer', _models.NoPhysics_KAN_Graphormer),
        # 层
        ('RobustKANLinear', _layers.RobustKANLinear),
        ('StandardMLP', _layers.StandardMLP),
        ('GatedTCN', _layers.GatedTCN),
        ('RBFSpatialEncoding', _layers.RBFSpatialEncoding),
        ('GraphormerLayer', _layers.GraphormerLayer),
        ('ImprovedGATLayer', _layers.ImprovedGATLayer),
        # 数据
        ('DataCollector', _data.DataCollector),
        ('CityLevelLogMinMaxScaler', _data.CityLevelLogMinMaxScaler),
        ('CityLevelLinearMinMaxScaler', _data.CityLevelLinearMinMaxScaler),
        ('TemporalFeatureEnhancer', _data.TemporalFeatureEnhancer),
        ('ImprovedTemporalDataset', _data.ImprovedTemporalDataset),
        # 损失
        ('CurriculumScientificLoss', _loss.CurriculumScientificLoss),
    ]
    for name, cls in _classes:
        setattr(__main__, name, cls)


_setup_backward_compat()


def load_results_from_dir(result_dir):
    """从结果目录加载所有模型的 pkl 结果"""
    results = {}
    results_dir = os.path.join(result_dir, "results")
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return results

    pkl_files = glob.glob(os.path.join(results_dir, "*_result.pkl"))
    for pkl_path in pkl_files:
        model_name = os.path.basename(pkl_path).replace("_result.pkl", "")
        try:
            with open(pkl_path, 'rb') as f:
                results[model_name] = pickle.load(f)
            print(f"  ✅ 加载: {model_name}")
        except Exception as e:
            print(f"  ❌ 加载失败 {model_name}: {e}")

    # 尝试加载 full_pred 数据 (可能内嵌在 pkl 中，也可能是独立文件)
    if 'Full_KAN_Graphormer' in results:
        fk = results['Full_KAN_Graphormer']
        if 'full_pred' not in fk:
            # 尝试独立 full_pred.pkl
            fp_path = os.path.join(result_dir, "results", "full_pred.pkl")
            if os.path.exists(fp_path):
                try:
                    with open(fp_path, 'rb') as f:
                        fk['full_pred'] = pickle.load(f)
                    print(f"  ✅ 加载: full_pred (独立文件)")
                except Exception as e:
                    print(f"  ⚠️ full_pred 加载失败: {e}")

    # 加载 summary.json (如果有)
    summary_path = os.path.join(result_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        # 提取 cities 和 dates
        cities = summary.get('cities', TARGET_CITIES)
        try:
            dates = [pd.Timestamp(d) for d in summary.get('dates', [])]
        except:
            dates = []
        return results, cities, dates, summary

    return results, TARGET_CITIES, [], {}


def plot_single_result_dir(result_dir, model_filter=None, comparison_only=False):
    """为单个结果目录生成所有图表"""
    print(f"\n读取结果目录: {result_dir}")
    results, cities, dates, summary = load_results_from_dir(result_dir)

    if not results:
        print("未找到任何结果文件!")
        return

    viz_manager = ImprovedActualDataVisualizer(result_dir, cities)

    # 构建封城索引
    lockdown_idx = {}
    if dates:
        for city, info in LOCKDOWN_INFO.items():
            if city in cities and info['start'] is not None and info['end'] is not None:
                try:
                    start_idx = dates.index(pd.Timestamp(info['start']))
                    end_idx = dates.index(pd.Timestamp(info['end']))
                    lockdown_idx[city] = {'start_idx': start_idx, 'end_idx': end_idx}
                except:
                    pass

    # 构建城市特征
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

    # 确定要处理的模型列表
    model_names = list(results.keys())
    if model_filter:
        if model_filter in model_names:
            model_names = [model_filter]
        else:
            print(f"模型 '{model_filter}' 不在结果中，可用: {model_names}")
            return

    if not comparison_only:
        for model_name in model_names:
            print(f"\n  正在为 {model_name} 生成图表...")
            result = results[model_name]

            train_data = {
                'training_history': result.get('train_history', {}),
                'model_parameters': result.get('collector_data', {}).get('model_parameters', {})
            }
            eval_data = result.get('eval_data', {})

            viz_manager.plot_actual_model_architecture(train_data, model_name)
            viz_manager.plot_actual_training_history(train_data, model_name)
            viz_manager.plot_actual_prediction_results(eval_data, model_name)
            viz_manager.plot_error_diagnostics(eval_data, model_name)

            # KAN splines (需要模型对象)
            model_info = result.get('model_info', {})
            if model_info.get('model') is not None:
                viz_manager.plot_kan_splines(model_info['model'], model_name)

            if model_info.get('has_physics', False):
                viz_manager.plot_actual_parameter_analysis(eval_data, model_name)
                viz_manager.plot_parameter_evolution(eval_data, model_name, dates, lockdown_idx)
                viz_manager.plot_epidemiological_insights(eval_data, model_name, LOCKDOWN_INFO)

            if 'Graphormer' in model_name or 'GAT' in model_name:
                viz_manager.plot_actual_attention_weights(eval_data, model_name)

            viz_manager.print_quantitative_insights(eval_data, city_features_df, model_name)
            viz_manager.generate_academic_thesis_report(eval_data, model_name, city_features_df)

    # 消融对比图 (始终生成，如果有多个模型)
    if len(results) >= 2:
        print("\n>>> 生成消融实验对比图...")
        viz_manager.plot_ablation_comparison(results)

    # 封城参数对比图 (由 visualizer 统一生成，与消融对比图风格一致)
    if 'Full_KAN_Graphormer' in results:
        if results['Full_KAN_Graphormer'].get('full_pred'):
            print("\n>>> 生成封城参数对比图...")
            viz_manager._plot_lockdown_params_comparison(results)

    print(f"\n✅ 图表生成完成! 输出目录: {result_dir}/visualizations/")


def plot_multi_seed_summary(result_dirs):
    """从多个种子目录生成汇总对比"""
    print(f"\n读取 {len(result_dirs)} 个种子目录...")

    all_metrics = defaultdict(list)

    for d in result_dirs:
        summary_path = os.path.join(d, "summary.json")
        if not os.path.exists(summary_path):
            print(f"  ⚠️ {d}: 无 summary.json")
            continue

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        for model_name, model_results in summary.get('results', {}).items():
            r2 = model_results.get('test_r2')
            rmse = model_results.get('test_rmse')
            if r2 is not None:
                all_metrics[model_name].append({'r2': r2, 'rmse': rmse})

    if not all_metrics:
        print("没有找到任何指标数据!")
        return

    # 计算均值 ± 标准差
    print("\n" + "=" * 70)
    print(f"{'模型':<30} {'R² (均值±σ)':<20} {'RMSE (均值±σ)':<20}")
    print("=" * 70)

    model_stats = {}
    for model_name, metrics_list in all_metrics.items():
        r2s = [m['r2'] for m in metrics_list if m['r2'] is not None]
        rmses = [m['rmse'] for m in metrics_list if m['rmse'] is not None]
        if r2s:
            model_stats[model_name] = {
                'r2_mean': np.mean(r2s), 'r2_std': np.std(r2s),
                'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses)
            }
            print(f"{model_name:<30} {np.mean(r2s):.4f}±{np.std(r2s):.4f}    "
                  f"{np.mean(rmses):.4f}±{np.std(rmses):.4f}")

    print("=" * 70)

    # 绘制多种子对比图
    print("\n>>> 生成多种子汇总图...")
    _plot_multi_seed_comparison(model_stats, result_dirs[0])


def _plot_multi_seed_comparison(model_stats, output_base_dir):
    import matplotlib.pyplot as plt

    output_dir = os.path.join(output_base_dir, "visualizations", "comparison")
    os.makedirs(output_dir, exist_ok=True)

    model_names = list(model_stats.keys())
    r2_means = [model_stats[m]['r2_mean'] for m in model_names]
    r2_stds = [model_stats[m]['r2_std'] for m in model_names]

    model_colors = {
        'MLP_Baseline': '#1f77b4', 'LSTM_Baseline': '#ff7f0e',
        'GAT_Baseline': '#2ca02c', 'M_Graphormer_Baseline': '#d62728',
        'KAN_Only': '#9467bd', 'NoPhysics_KAN_Graphormer': '#8c564b',
        'Full_KAN_Graphormer': '#e377c2',
    }
    colors = [model_colors.get(n, 'gray') for n in model_names]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(model_names))
    bars = ax.bar(x, r2_means, yerr=r2_stds, color=colors, alpha=0.8,
                  capsize=5, error_kw={'linewidth': 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r'$R^2$ (Mean ± Std)', fontsize=12)
    ax.set_title('Multi-Seed Ablation Study: ' + r'$R^2$ Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean_val, std_val in zip(bars, r2_means, r2_stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + std_val + 0.005,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "multi_seed_r2_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  多种子对比图已保存: {out_path}")


def find_latest_result_dir(base_dir=None):
    """自动找到最新的消融实验结果目录"""
    if base_dir is None:
        base_dir = r"E:\Claude code\KAN+\result"
    pattern = os.path.join(base_dir, "ablation_study_*")
    dirs = sorted(glob.glob(pattern), reverse=True)
    if not dirs:
        return None
    # 过滤掉不含 results/ 子目录的
    for d in dirs:
        if os.path.isdir(os.path.join(d, "results")):
            return d
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='独立画图工具 — 从 pkl 结果生成图表')
    parser.add_argument('--dir', type=str, default=None,
                        help='单个结果目录路径 (默认: 自动选最新)')
    parser.add_argument('--dirs', type=str, nargs='+', default=None,
                        help='多个结果目录路径 (用于多种子汇总)')
    parser.add_argument('--model', type=str, default=None,
                        help='只为此模型生成图表 (默认: 全部模型)')
    parser.add_argument('--comparison-only', action='store_true',
                        help='只生成消融对比图，跳过单模型图表')
    parser.add_argument('--summary', action='store_true',
                        help='生成多种子汇总对比')
    args = parser.parse_args()

    if args.dirs and args.summary:
        plot_multi_seed_summary(args.dirs)
    elif args.dir:
        plot_single_result_dir(args.dir, args.model, args.comparison_only)
    elif not args.dirs:
        # 自动找最新结果目录
        latest = find_latest_result_dir()
        if latest:
            print(f"自动选择最新结果: {latest}")
            plot_single_result_dir(latest, args.model, args.comparison_only)
        else:
            print("未找到任何消融实验结果目录，请用 --dir 指定路径")
            print("\n示例:")
            print("  python plot_results.py --dir ../result/ablation_study_v9_20260609_151421")
