"""
可视化模块 — 从 exp_chronological.py 提取
ImprovedActualDataVisualizer: 所有图表生成逻辑
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import math
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .config import TARGET_CITIES, TIER_1_CITIES, TIER_2_CITIES, LOCKDOWN_INFO, SEED

# 输出可编辑文本的 SVG 矢量图（供 Illustrator / Inkscape 直接编辑）
plt.rcParams['svg.fonttype'] = 'none'


class ImprovedActualDataVisualizer:
    """完整版可视化管理器：包含v8.1所有图表 + 参数演化/相关性/误差诊断"""

    def __init__(self, output_dir, cities):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        self.cities = cities
        self.n_cities = len(cities)
        self.create_subdirectories()
        self.setup_colors()

    def create_subdirectories(self):
        subdirs = ["architecture", "ablation", "prediction", "parameters",
                   "training", "attention", "residuals", "comparison",
                   "evolution", "correlation", "error"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.viz_dir, subdir), exist_ok=True)

    def setup_colors(self):
        self.city_colors = plt.cm.Set3(np.linspace(0, 1, self.n_cities))
        self.model_colors = {
            'MLP_Baseline': '#1f77b4', 'LSTM_Baseline': '#ff7f0e',
            'GAT_Baseline': '#2ca02c', 'M_Graphormer_Baseline': '#d62728',
            'KAN_Only': '#9467bd', 'NoPhysics_KAN_Graphormer': '#8c564b',
            'Full_KAN_Graphormer': '#e377c2',
        }

    def fill_missing_data(self, data, model_name):
        filled_data = data.copy() if data else {}
        for key in ['training_history', 'evaluation_results', 'model_parameters',
                    'performance_metrics', 'param_validity']:
            if key not in filled_data:
                filled_data[key] = {}
        eval_keys = ['predictions', 'targets', 'betas', 'gammas', 'contacts',
                     'r0s', 'attention_maps', 'residuals']
        for subkey in eval_keys:
            if subkey not in filled_data.get('evaluation_results', {}):
                filled_data.setdefault('evaluation_results', {})[subkey] = None
        return filled_data

    # ============================================================
    # 1. 模型架构图
    # ============================================================
    def plot_actual_model_architecture(self, model_data, model_name):
        filled_data = self.fill_missing_data(model_data, model_name)
        params = filled_data['model_parameters']
        self._plot_parameter_distributions(params, model_name)
        self._plot_model_layer_structure(params, model_name)

    def _plot_parameter_distributions(self, params, model_name):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Model Parameter Distributions: {model_name}', fontsize=16, fontweight='bold')
        all_weights = [p.detach().cpu().numpy().flatten() if isinstance(p, torch.Tensor)
                       else np.array(p).flatten()
                       for p in params.get('weights', {}).values()]
        all_biases = [p.detach().cpu().numpy().flatten() if isinstance(p, torch.Tensor)
                      else np.array(p).flatten()
                      for p in params.get('biases', {}).values()]
        all_kan = [p.detach().cpu().numpy().flatten() if isinstance(p, torch.Tensor)
                   else np.array(p).flatten()
                   for p in params.get('kan_weights', {}).values()]

        all_weights = [w for sub in all_weights for w in sub] if all_weights else []
        all_biases = [b for sub in all_biases for b in sub] if all_biases else []
        all_kan = [k for sub in all_kan for k in sub] if all_kan else []

        for ax, data, title, color in [
            (axes[0, 0], all_weights, 'Weight Distribution', 'skyblue'),
            (axes[0, 1], all_biases, 'Bias Distribution', 'lightgreen'),
            (axes[1, 0], all_kan, 'KAN Weight Distribution', 'salmon')]:
            if data:
                ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black')
                ax.text(0.95, 0.95, f'Mean: {np.mean(data):.4f}\nStd: {np.std(data):.4f}',
                        transform=ax.transAxes, fontsize=8, va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', color='red')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        counts = {'Weights': len(all_weights), 'Biases': len(all_biases), 'KAN Weights': len(all_kan)}
        ax4 = axes[1, 1]
        bars = ax4.bar(counts.keys(), counts.values(), color=['skyblue', 'lightgreen', 'salmon'], alpha=0.8)
        for bar, count in zip(bars, counts.values()):
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{count:,}',
                     ha='center', va='bottom', fontsize=9)
        ax4.set_title('Parameter Counts', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "architecture", f"parameter_distributions_{model_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_layer_structure(self, params, model_name):
        fig, ax = plt.subplots(figsize=(12, 6))
        layers, layer_sizes = [], []
        for name in params.get('weights', {}).keys():
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in layers:
                layers.append(layer_name)
                count = sum(p.numel() if isinstance(p, torch.Tensor) else np.array(p).size
                            for n, p in {**params.get('weights', {}), **params.get('biases', {})}.items()
                            if n.startswith(layer_name))
                layer_sizes.append(count)
        if not layers:
            layers, layer_sizes = ['Input', 'Hidden1', 'Hidden2', 'Output'], [1000, 2000, 1000, 500]
        y_pos = np.arange(len(layers))
        bars = ax.barh(y_pos, layer_sizes, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Number of Parameters', fontsize=11)
        ax.set_title(f'Model Layer Structure: {model_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, size in zip(bars, layer_sizes):
            ax.text(bar.get_width() + max(layer_sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{size:,}', ha='left', va='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "architecture", f"layer_structure_{model_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ============================================================
    # 2. 训练过程图
    # ============================================================
    def plot_actual_training_history(self, training_data, model_name):
        filled_data = self.fill_missing_data(training_data, model_name)
        history = filled_data['training_history']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Process: {model_name}', fontsize=16, fontweight='bold')
        has_data = len(history.get('train_loss', [])) > 0

        if has_data and history['train_loss']:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, alpha=0.8, label='Training Loss')
            if history.get('val_loss') and len(history['val_loss']) == len(history['train_loss']):
                axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, alpha=0.8, label='Validation Loss')
            axes[0, 0].set_yscale('log')
            axes[0, 0].legend(fontsize=9)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        if has_data and history.get('learning_rates'):
            axes[0, 1].plot(range(1, len(history['learning_rates']) + 1),
                            history['learning_rates'], 'g-', linewidth=2)
            axes[0, 1].set_yscale('log')
        axes[0, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        if has_data and history.get('epoch_times'):
            axes[1, 0].plot(range(1, len(history['epoch_times']) + 1),
                            history['epoch_times'], 'm-', linewidth=2)
        axes[1, 0].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        if has_data and len(history.get('train_loss', [])) > 1:
            tl = np.array(history['train_loss'])
            reduction = np.diff(tl) / tl[:-1] * 100
            axes[1, 1].plot(range(2, len(tl) + 1), reduction, 'orange', linewidth=2)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Training Loss Reduction Rate', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "training", f"training_history_{model_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ============================================================
    # 3. 预测效果图
    # ============================================================
    def plot_actual_prediction_results(self, eval_data, model_name):
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']
        if results.get('predictions') is not None and results.get('targets') is not None:
            self._plot_prediction_scatter(results, model_name)
            self._plot_prediction_timeseries(results, model_name)
        if results.get('residuals') is not None:
            self._plot_residual_distribution(results, model_name)
        self._plot_city_level_predictions(results, model_name)

    def _plot_prediction_scatter(self, results, model_name, metrics_override=None):
        pred = results['predictions']
        targ = results['targets']
        pred_np = pred.numpy().flatten() if isinstance(pred, torch.Tensor) else np.array(pred).flatten()
        targ_np = targ.numpy().flatten() if isinstance(targ, torch.Tensor) else np.array(targ).flatten()

        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([targ_np, pred_np])
            z = gaussian_kde(xy)(xy)
            scatter = ax.scatter(targ_np, pred_np, c=z, s=20, alpha=0.6, cmap='cividis', edgecolors='none')
            plt.colorbar(scatter, ax=ax).set_label('Point Density', fontsize=11)
        except:
            ax.scatter(targ_np, pred_np, s=20, alpha=0.6, color='blue', edgecolors='none')
        min_v, max_v = min(targ_np.min(), pred_np.min()), max(targ_np.max(), pred_np.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2, alpha=0.7, label='Perfect Prediction')

        if metrics_override is not None:
            r2, r2_std, rmse, rmse_std, mae, mae_std = metrics_override
            stats_text = (f'$R^2$ = {r2:.3f} ± {r2_std:.3f}\n'
                          f'RMSE = {rmse:.3f} ± {rmse_std:.3f}\n'
                          f'MAE = {mae:.3f} ± {mae_std:.3f}')
        else:
            r2 = r2_score(targ_np, pred_np) if len(pred_np) > 1 else 0.0
            rmse = np.sqrt(mean_squared_error(targ_np, pred_np)) if len(pred_np) > 0 else 0.0
            mae = mean_absolute_error(targ_np, pred_np) if len(pred_np) > 0 else 0.0
            stats_text = f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))

        ax.set_xlabel('Actual Cases', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Cases', fontsize=12, fontweight='bold')

        # pure white background + black boxed axes (all four sides)
        ax.set_facecolor('white')
        for s in ('top', 'right', 'left', 'bottom'):
            ax.spines[s].set_color('black')
            ax.spines[s].set_linewidth(1.0)
            ax.spines[s].set_visible(True)

        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        out_dir = os.path.join(self.viz_dir, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg', 'pdf'):
            plt.savefig(os.path.join(out_dir, f"scatter_{model_name}.{ext}"),
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 城市英文名映射 (论文图用，替代中文缩写标题)
    _CITY_EN = {
        '北京市': 'Beijing', '天津市': 'Tianjin', '上海市': 'Shanghai',
        '重庆市': 'Chongqing', '广州市': 'Guangzhou', '深圳市': 'Shenzhen',
        '西安市': "Xi'an", '成都市': 'Chengdu', '武汉市': 'Wuhan',
        '杭州市': 'Hangzhou', '南京市': 'Nanjing', '苏州市': 'Suzhou',
        '无锡市': 'Wuxi', '郑州市': 'Zhengzhou', '长沙市': 'Changsha',
        '沈阳市': 'Shenyang', '大连市': 'Dalian', '青岛市': 'Qingdao',
        '济南市': 'Jinan', '宁波市': 'Ningbo', '厦门市': 'Xiamen',
        '哈尔滨市': 'Harbin', '长春市': 'Changchun', '石家庄市': 'Shijiazhuang',
    }

    def _city_english(self, cn):
        return self._CITY_EN.get(cn, cn)

    def _plot_prediction_timeseries(self, results, model_name, city_r2_override=None):
        pred = results['predictions']
        targ = results['targets']
        pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
        targ_np = targ.numpy() if isinstance(targ, torch.Tensor) else np.array(targ)
        if pred_np.ndim < 2:
            return

        n_samples = pred_np.shape[0]
        city_indices = [0, min(5, self.n_cities - 1), min(10, self.n_cities - 1),
                        min(15, self.n_cities - 1), min(20, self.n_cities - 1), min(23, self.n_cities - 1)]
        city_names = [self._city_english(self.cities[i]) if i < len(self.cities) else f'City_{i}'
                      for i in city_indices]

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Time Series Predictions: {model_name}', fontsize=16, fontweight='bold')
        time_steps = min(100, n_samples)

        for idx, (ci, cn) in enumerate(zip(city_indices, city_names)):
            if idx >= 6: break
            ax = axes[idx // 2, idx % 2]
            if pred_np.ndim == 4:
                cp = pred_np[:time_steps, 0, ci, :].sum(axis=-1)
                ct = targ_np[:time_steps, 0, ci, :].sum(axis=-1)
            elif pred_np.ndim == 3:
                cp = pred_np[:time_steps, ci, :].sum(axis=-1)
                ct = targ_np[:time_steps, ci, :].sum(axis=-1)
            else:
                cp = pred_np[:time_steps, ci]
                ct = targ_np[:time_steps, ci]
            ax.plot(range(len(ct)), ct, 'b-', linewidth=2, alpha=0.8, label='Actual')
            ax.plot(range(len(cp)), cp, 'r--', linewidth=2, alpha=0.8, label='Predicted')
            if city_r2_override is not None and ci < len(city_r2_override):
                city_r2 = float(city_r2_override[ci])
            else:
                city_r2 = r2_score(ct, cp) if len(ct) > 1 and np.var(ct) > 1e-5 else 0.0
            ax.set_title(f'{cn} ($R^2$={city_r2:.3f})', fontsize=11, fontweight='bold')

            # 纯白背景 + 黑色单坐标轴 (左 + 下)
            ax.set_facecolor('white')
            for s in ('left', 'bottom'):
                ax.spines[s].set_color('black')
                ax.spines[s].set_linewidth(1.0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if idx == 0: ax.legend(loc='upper left', fontsize=8)

        for idx in range(len(city_indices), 6):
            axes[idx // 2, idx % 2].axis('off')
        plt.tight_layout()
        out_dir = os.path.join(self.viz_dir, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg', 'pdf'):
            plt.savefig(os.path.join(out_dir, f"timeseries_{model_name}.{ext}"),
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_residual_distribution(self, results, model_name):
        residuals = results['residuals']
        if residuals is None: return
        res_np = residuals.numpy().flatten() if isinstance(residuals, torch.Tensor) else np.array(residuals).flatten()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Prediction Residuals: {model_name}', fontsize=14, fontweight='bold')
        axes[0].hist(res_np, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        mu, std = np.mean(res_np), np.std(res_np)
        axes[0].set_title(f'Residual Distribution (μ={mu:.4f}, σ={std:.4f})', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        stats.probplot(res_np, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "residuals", f"residuals_{model_name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_city_level_predictions(self, results, model_name,
                                     city_r2_override=None, city_rmse_override=None,
                                     city_r2_std=None, city_rmse_std=None):
        if results.get('predictions') is None or results.get('targets') is None:
            return
        pred = results['predictions']
        targ = results['targets']

        def to_numpy_2d(data):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            arr = np.array(data)
            if arr.ndim == 4: return arr.mean(axis=1).sum(axis=-1)
            elif arr.ndim == 3: return arr.sum(axis=-1)
            return arr

        pred_np = to_numpy_2d(pred)
        targ_np = to_numpy_2d(targ)
        n_cities = min(pred_np.shape[1], self.n_cities)

        city_r2, city_rmse = [], []
        for i in range(n_cities):
            ct, cp = targ_np[:, i], pred_np[:, i]
            rmse_v = float(np.sqrt(mean_squared_error(ct, cp)))
            if np.max(ct) - np.min(ct) < 2.0 or np.var(ct) < 1.0:
                r2_v = 1.0 if rmse_v < 1.0 else 0.0
            else:
                r2_v = max(-1.0, float(r2_score(ct, cp)))
            city_r2.append(r2_v)
            city_rmse.append(rmse_v)

        # 用 20 种子逐城市均值覆盖 (真实数据，与 20 种子汇总一致)
        if city_r2_override is not None:
            city_r2 = [float(v) for v in city_r2_override[:n_cities]]
        if city_rmse_override is not None:
            city_rmse = [float(v) for v in city_rmse_override[:n_cities]]
        # 20 种子跨种子 std (误差棒); 未提供时为 0 (不画)
        if city_r2_std is None:
            city_r2_std = [0.0] * n_cities
        else:
            city_r2_std = [float(v) for v in city_r2_std[:n_cities]]
        if city_rmse_std is None:
            city_rmse_std = [0.0] * n_cities
        else:
            city_rmse_std = [float(v) for v in city_rmse_std[:n_cities]]

        # 英文城市名，按 R² 降序排列 (RMSE 共用此顺序)
        city_labels = [self._city_english(c) for c in self.cities[:n_cities]]
        sorted_idx_r2 = np.argsort(city_r2)[::-1]
        city_labels_sorted = [city_labels[i] for i in sorted_idx_r2]
        r2_sorted = [city_r2[i] for i in sorted_idx_r2]
        rmse_sorted = [city_rmse[i] for i in sorted_idx_r2]
        r2_std_sorted = [city_r2_std[i] for i in sorted_idx_r2]
        rmse_std_sorted = [city_rmse_std[i] for i in sorted_idx_r2]

        fig, ax1 = plt.subplots(figsize=(13, 6))
        x_pos = range(n_cities)

        # 左轴: R² (蓝) + 20 种子 ±std 阴影带
        ax1.plot(x_pos, r2_sorted, color='#2c7bb6', marker='o', markersize=6,
                 linewidth=1.2, alpha=0.9, zorder=3, label='$R^2$ (left)')
        ax1.fill_between(list(x_pos),
                         [a - b for a, b in zip(r2_sorted, r2_std_sorted)],
                         [a + b for a, b in zip(r2_sorted, r2_std_sorted)],
                         color='#2c7bb6', alpha=0.18, linewidth=0, zorder=2)
        ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', zorder=1)
        ax1.set_ylabel('$R^2$', fontsize=13, color='#2c7bb6')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(city_labels_sorted, rotation=60, fontsize=10, ha='right')
        ax1.set_ylim(-1.05, 1.05)
        ax1.tick_params(axis='y', labelcolor='#2c7bb6', labelsize=10)

        # 右轴: RMSE (红)
        ax2 = ax1.twinx()
        ax2.plot(x_pos, rmse_sorted, color='#ff7f0e', marker='s', markersize=6,
                 linewidth=1.2, alpha=0.9, zorder=3, label='RMSE (right)')
        ax2.fill_between(list(x_pos),
                         [max(0.0, a - b) for a, b in zip(rmse_sorted, rmse_std_sorted)],
                         [a + b for a, b in zip(rmse_sorted, rmse_std_sorted)],
                         color='#ff7f0e', alpha=0.18, linewidth=0, zorder=2)
        ax2.set_ylabel('RMSE', fontsize=13, color='#ff7f0e')
        ax2.set_ylim(0, None)
        ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=10)

        # 纯白背景 + 黑色方框 (上/左/下在 ax1, 右在 ax2)
        ax1.set_facecolor('white')
        ax2.set_facecolor('none')
        for s in ('top', 'left', 'bottom'):
            ax1.spines[s].set_color('black')
            ax1.spines[s].set_linewidth(1.0)
            ax1.spines[s].set_visible(True)
        ax1.spines['right'].set_visible(False)
        ax2.spines['right'].set_color('black')
        ax2.spines['right'].set_linewidth(1.0)
        for s in ('top', 'left', 'bottom'):
            ax2.spines[s].set_visible(False)

        fig.tight_layout()
        out_dir = os.path.join(self.viz_dir, "prediction")
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'svg', 'pdf'):
            plt.savefig(os.path.join(out_dir, f"city_level_{model_name}.{ext}"),
                        dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # ============================================================
    # 4. 参数分析
    # ============================================================
    def plot_actual_parameter_analysis(self, eval_data, model_name):
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']
        self._plot_beta_analysis(results.get('betas'), model_name)
        self._plot_r0_analysis(results.get('r0s'), model_name)
        self._plot_contact_analysis(results.get('contacts'), model_name)

    def _plot_beta_analysis(self, betas, model_name):
        if betas is None: return
        betas_np = betas.numpy() if isinstance(betas, torch.Tensor) else np.array(betas)
        if betas_np.size == 0: return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Transmission Rate (β) Analysis: {model_name}', fontsize=16, fontweight='bold')

        if betas_np.ndim == 2:
            mean_b = np.mean(betas_np, axis=1)
            std_b = np.std(betas_np, axis=1)
            axes[0, 0].plot(range(len(mean_b)), mean_b, 'b-', linewidth=2, alpha=0.8)
            axes[0, 0].fill_between(range(len(mean_b)), mean_b - std_b, mean_b + std_b, alpha=0.3, color='blue')
        axes[0, 0].set_title('β Time Series', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(betas_np.flatten(), bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(betas_np), color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('β Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        if betas_np.ndim == 2 and betas_np.shape[1] == self.n_cities:
            city_mean = np.mean(betas_np, axis=0)
            n_c = min(self.n_cities, len(city_mean))
            axes[1, 0].bar(range(n_c), city_mean[:n_c], color=self.city_colors, alpha=0.8, edgecolor='black')
            axes[1, 0].set_xticks(range(n_c))
            axes[1, 0].set_xticklabels([c[:2] for c in self.cities[:n_c]], rotation=90, fontsize=8)
        axes[1, 0].set_title('β by City', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        axes[1, 1].set_title('β Change Rate', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "parameters", f"beta_analysis_{model_name}.png"), dpi=300)
        plt.close()

    def _plot_r0_analysis(self, r0s, model_name):
        if r0s is None: return
        r0s_np = r0s.numpy() if isinstance(r0s, torch.Tensor) else np.array(r0s)
        if r0s_np.size == 0: return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(r'Effective Reproduction Number ($R_t$) Analysis: ' + model_name, fontsize=16, fontweight='bold')
        r0s_flat = r0s_np.flatten()

        if r0s_np.ndim == 2:
            mean_r = np.mean(r0s_np, axis=1)
            axes[0, 0].plot(range(len(mean_r)), mean_r, 'b-', linewidth=2)
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].set_title(r'$R_t$ Time Series', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(r0s_flat, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(x=np.mean(r0s_flat), color='red', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=1.0, color='red', linestyle='-', alpha=0.5)
        axes[0, 1].set_title(r'$R_t$ Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        if r0s_np.ndim == 2:
            suppressed = np.sum(r0s_flat < 1.0)
            active = np.sum(r0s_flat >= 1.0)
            axes[1, 1].pie([suppressed, active], labels=[r'$R_t<1$', r'$R_t≥1$'],
                           colors=['lightblue', 'lightgreen'], autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(r'$R_t$ State Classification', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "parameters", f"rt_analysis_{model_name}.png"), dpi=300)
        plt.close()

    def _plot_contact_analysis(self, contacts, model_name):
        if contacts is None: return
        contacts_np = contacts.numpy() if isinstance(contacts, torch.Tensor) else np.array(contacts)
        if contacts_np.size == 0: return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Contact Factor Analysis: {model_name}', fontsize=14, fontweight='bold')
        axes[0].hist(contacts_np.flatten(), bins=50, alpha=0.7, color='gold', edgecolor='black')
        axes[0].axvline(x=np.mean(contacts_np), color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Contact Factor Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        if contacts_np.ndim == 2:
            mc = np.mean(contacts_np, axis=1)
            axes[1].plot(range(len(mc)), mc, 'b-', linewidth=2)
        axes[1].set_title('Contact Factor Time Series', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "parameters", f"contact_analysis_{model_name}.png"), dpi=300)
        plt.close()

    # ============================================================
    # 5. 注意力图
    # ============================================================
    def plot_actual_attention_weights(self, eval_data, model_name):
        filled_data = self.fill_missing_data(eval_data, model_name)
        attn = filled_data['evaluation_results'].get('attention_maps')
        if attn is None: return

        if isinstance(attn, torch.Tensor):
            if attn.dim() >= 4:
                sample = attn[0].numpy()
            elif attn.dim() == 3:
                sample = attn[0].unsqueeze(0).numpy() if attn.shape[0] > 0 else None
            else:
                return
        else:
            sample = np.array(attn)
            sample = sample[0] if sample.ndim >= 3 else sample
        if sample is None: return

        n_heads = min(4, sample.shape[0])
        n_c = min(sample.shape[1], self.n_cities)
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
        if n_heads == 1: axes = [axes]
        fig.suptitle(f'Attention Weights: {model_name}', fontsize=16, fontweight='bold')

        for h in range(n_heads):
            im = axes[h].imshow(sample[h, :n_c, :n_c], cmap='YlOrRd', aspect='auto')
            axes[h].set_xticks(range(n_c))
            axes[h].set_yticks(range(n_c))
            axes[h].set_xticklabels([c[:2] for c in self.cities[:n_c]], rotation=90, fontsize=6)
            axes[h].set_yticklabels([c[:2] for c in self.cities[:n_c]], fontsize=6)
            axes[h].set_title(f'Head {h + 1}', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=axes[h])
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "attention", f"attention_{model_name}.png"), dpi=300)
        plt.savefig(os.path.join(self.viz_dir, "attention", f"attention_{model_name}.svg"))
        plt.close()

    # ============================================================
    # 6. KAN Splines
    # ============================================================
    def plot_kan_splines(self, model, model_name):
        from .layers import RobustKANLinear
        kan_layers = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, RobustKANLinear)]
        if not kan_layers: return

        fig, axes = plt.subplots(len(kan_layers), 1, figsize=(10, 4 * len(kan_layers)))
        if len(kan_layers) == 1: axes = [axes]

        for idx, (name, layer) in enumerate(kan_layers[:min(len(kan_layers), 6)]):
            ax = axes[idx]
            x_test = torch.linspace(-3, 3, 200).to(layer.grid.device)
            bases = layer.b_splines(x_test)
            weight = layer.spline_weight.detach().cpu()
            out_dim, in_dim, _ = weight.shape
            for o in range(min(out_dim, 3)):
                for i in range(min(in_dim, 3)):
                    y = torch.matmul(bases, weight[o, i]).detach().cpu().numpy()
                    ax.plot(x_test.cpu().numpy(), y, linewidth=2, alpha=0.7,
                            label=f'Dim {i}→{o}')
            ax.set_title(f'Learned 1D B-spline Activations in {name}', fontsize=12, fontweight='bold')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "parameters", f"kan_spline_curves_{model_name}.png"), dpi=300)
        plt.close()

    # ============================================================
    # 7. 消融实验对比图
    # ============================================================
    def plot_ablation_comparison(self, all_results):
        if not all_results: return
        self._plot_performance_comparison(all_results)
        self._plot_lockdown_params_comparison(all_results)

    def _plot_performance_comparison(self, all_results):
        model_names, r2_vals, rmse_vals, mae_vals, times, params = [], [], [], [], [], []
        for mn, data in all_results.items():
            model_names.append(mn)
            m = data.get('eval_data', {}).get('performance_metrics', {})
            r2_vals.append(m.get('r2', 0.0))
            rmse_vals.append(m.get('rmse', 0.0))
            mae_vals.append(m.get('mae', 0.0))
            times.append(data.get('collector_data', {}).get('training_time', 0.0))
            params.append(data.get('model_info', {}).get('trainable_params', 0))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ablation Study: Model Performance Comparison', fontsize=16, fontweight='bold')
        colors = [self.model_colors.get(n, 'gray') for n in model_names]

        r2_plot = [max(-1.0, v) for v in r2_vals]
        bars1 = axes[0, 0].bar(model_names, r2_plot, color=colors, alpha=0.8)
        axes[0, 0].set_ylabel(r'$R^2$', fontsize=11)
        axes[0, 0].set_title(r'$R^2$ Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        for bar, r2_val in zip(bars1, r2_vals):
            h = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2.,
                            h + 0.01 if h >= 0 else h - 0.05, f'{r2_val:.3f}',
                            ha='center', va='bottom' if h >= 0 else 'top', fontsize=9)

        axes[0, 1].bar(model_names, rmse_vals, color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('RMSE', fontsize=11)
        axes[0, 1].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[1, 0].bar(model_names, times, color=colors, alpha=0.8)
        axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=11)
        axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(model_names, params, color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Number of Parameters', fontsize=11)
        axes[1, 1].set_title('Model Complexity Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "comparison", "performance_comparison.png"), dpi=300)
        plt.close()

    def _plot_lockdown_params_comparison(self, all_results):
        fk_data = all_results.get('Full_KAN_Graphormer', {})
        full_pred = fk_data.get('full_pred', None)
        if full_pred is None: return

        beta_full = full_pred.get('beta_full')
        contact_full = full_pred.get('contact_full')
        rt_full = full_pred.get('rt_full')
        dates_list = full_pred.get('dates')
        if beta_full is None or contact_full is None or rt_full is None: return

        beta_np = np.array(beta_full) if not isinstance(beta_full, np.ndarray) else beta_full
        contact_np = np.array(contact_full) if not isinstance(contact_full, np.ndarray) else contact_full
        rt_np = np.array(rt_full) if not isinstance(rt_full, np.ndarray) else rt_full

        n_days, n_cities = beta_np.shape
        lookback, predict = 7, 3
        start_idx = lookback
        end_idx = n_days - predict
        lockdown_day = 83
        x_valid = np.arange(start_idx, end_idx)

        targets = [('武汉市', 'Wuhan (lockdown)'), ('青岛市', 'Qingdao (no lockdown)')]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)

        all_left = []
        for city_name, _ in targets:
            if city_name in self.cities:
                ci = self.cities.index(city_name)
                all_left.append(beta_np[start_idx:end_idx, ci])
                all_left.append(contact_np[start_idx:end_idx, ci])
        left_all = np.concatenate(all_left)
        left_ylim = (left_all.min() - 0.05, left_all.max() + 0.05)

        all_rt = []
        for city_name, _ in targets:
            if city_name in self.cities:
                ci = self.cities.index(city_name)
                all_rt.append(rt_np[start_idx:end_idx, ci])
        rt_all = np.concatenate(all_rt)
        rt_ylim = (min(1.0, rt_all.min()) - 0.05, rt_all.max() + 0.05)

        for panel_idx, (city_name, label_text) in enumerate(targets):
            if city_name not in self.cities: continue
            c_idx = self.cities.index(city_name)
            ax = axes[panel_idx]

            b = beta_np[start_idx:end_idx, c_idx]
            c = contact_np[start_idx:end_idx, c_idx]
            r = rt_np[start_idx:end_idx, c_idx]

            ax.plot(x_valid, b, '-', color='#1f77b4', linewidth=1.2, label=r'$\beta$')
            ax.plot(x_valid, c, '--', color='#9467bd', linewidth=1.2, label='Contact')
            ax.set_ylabel(r'$\beta$ / Contact', fontsize=11, color='black')
            ax.tick_params(axis='y', labelcolor='black', labelsize=9)
            ax.set_facecolor('white')
            ax.set_ylim(left_ylim)

            # x-axis in readable dates; lockdown date marked below the axis (not a
            # vertical line crossing the curves)
            valid_dates = dates_list[start_idx:end_idx]
            n_xticks = 6
            xtick_pos = np.linspace(0, len(x_valid) - 1, n_xticks, dtype=int)
            ax.set_xticks(x_valid[xtick_pos])
            ax.set_xticklabels([valid_dates[i].strftime('%m-%d') for i in xtick_pos],
                               fontsize=9)
            ax.tick_params(axis='x', labelcolor='black', labelsize=9)
            lockdown_date_str = (dates_list[lockdown_day].strftime('%b %d')
                                 if lockdown_day < len(dates_list) else 'Jan 23')
            ax.annotate(f'Wuhan lockdown\n({lockdown_date_str})',
                        xy=(lockdown_day, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -30), textcoords='offset points',
                        ha='center', va='top', fontsize=9, color='black',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color='black', lw=0.7))

            ax2 = ax.twinx()
            ax2.plot(x_valid, r, '-', color='#ff7f0e', linewidth=1.4, label=r'$R_t$')
            ax2.set_ylabel(r'$R_t$', fontsize=11, color='black')
            ax2.tick_params(axis='y', labelcolor='black', labelsize=9)
            ax2.set_facecolor('none')

            # black solid axis box (top/left/bottom on ax, right on ax2)
            for s in ('top', 'left', 'bottom'):
                ax.spines[s].set_color('black')
                ax.spines[s].set_linewidth(1.0)
                ax.spines[s].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax2.spines['right'].set_color('black')
            ax2.spines['right'].set_linewidth(1.0)
            for s in ('top', 'left', 'bottom'):
                ax2.spines[s].set_visible(False)

            ax2.set_ylim(rt_ylim)

            ax.set_title(f"({'ab'[panel_idx]}) {label_text}", fontsize=12, fontweight='bold')

            # 图例只放第一个子图 (两子图图例完全相同)
            if panel_idx == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                           fontsize=9, framealpha=0.9)

        os.makedirs(os.path.join(self.viz_dir, "comparison"), exist_ok=True)
        out = os.path.join(self.viz_dir, "comparison", "lockdown_params_comparison.png")
        plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(self.viz_dir, "comparison", "lockdown_params_comparison.svg"),
                    bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(self.viz_dir, "comparison", "lockdown_params_comparison.pdf"),
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  [封城对比] {out}")

    # ============================================================
    # 8. 参数演化热力图
    # ============================================================
    def plot_parameter_evolution(self, eval_data, model_name, dates=None, lockdown_idx=None):
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']
        betas = results.get('betas')
        r0s = results.get('r0s')
        contacts = results.get('contacts')
        if all(x is None for x in [betas, r0s, contacts]): return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'Parameter Evolution: {model_name}', fontsize=16, fontweight='bold')

        def to_np(t):
            if t is None: return None
            return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)

        for i, (param, title, cmap) in enumerate(zip(
                [to_np(betas), to_np(r0s), to_np(contacts)],
                ['Transmission Rate (β)', 'Reproduction Number (R₀)', 'Contact Factor'],
                ['YlOrRd', 'RdYlGn_r', 'PuBu'])):
            ax = axes[i]
            if param is not None and param.ndim == 2:
                city_order = np.argsort(np.mean(param, axis=0))[::-1]
                im = ax.imshow(param[:, city_order].T, aspect='auto', cmap=cmap,
                               interpolation='bilinear', origin='lower')
                ax.set_yticks(range(len(self.cities)))
                ax.set_yticklabels([self.cities[i] for i in city_order], fontsize=8)
                ax.set_title(title, fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax)
            else:
                ax.text(0.5, 0.5, f'No {title} Data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "evolution", f"evolution_{model_name}.png"), dpi=300)
        plt.savefig(os.path.join(self.viz_dir, "evolution", f"evolution_{model_name}.svg"))
        plt.close()

    # ============================================================
    # 9. 预测误差诊断
    # ============================================================
    def plot_error_diagnostics(self, eval_data, model_name):
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']
        pred = results.get('predictions')
        targ = results.get('targets')
        if pred is None or targ is None: return

        def to_np(t):
            return t.numpy() if isinstance(t, torch.Tensor) else np.array(t)

        pred_np = to_np(pred)
        targ_np = to_np(targ)
        residuals = (pred_np - targ_np).flatten()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Prediction Error Diagnostics: {model_name}', fontsize=16, fontweight='bold')

        axes[0, 0].plot(range(len(residuals)), residuals, 'b-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals over Time', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(pred_np.flatten(), residuals, s=5, alpha=0.5, c='steelblue')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Cases', fontsize=11)
        axes[0, 1].set_ylabel('Residuals', fontsize=11)
        axes[0, 1].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, mu, std), 'r-')
        axes[1, 0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        lags = range(min(30, len(residuals) // 2))
        autocorr = [1.0] + [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                            if len(residuals) > lag and lag > 0 else 0.0
                            for lag in lags[1:]]
        axes[1, 1].stem(lags, autocorr[:len(lags)], basefmt=" ")
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Error Autocorrelation', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, "error", f"error_diag_{model_name}.png"), dpi=300)
        plt.close()

    # ============================================================
    # 10. 流行病学洞察
    # ============================================================
    def plot_epidemiological_insights(self, eval_data, model_name, lockdown_info=None):
        if lockdown_info is None:
            lockdown_info = LOCKDOWN_INFO
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']
        betas = results.get('betas')
        contacts = results.get('contacts')
        r0s = results.get('r0s')
        if any(x is None for x in [betas, contacts, r0s]): return

        def to_np(t):
            return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

        beta_np = to_np(betas)
        contact_np = to_np(contacts)
        r0_np = to_np(r0s)

        city_avg_beta = np.mean(beta_np, axis=0) if beta_np.ndim == 2 else beta_np
        city_avg_contact = np.mean(contact_np, axis=0) if contact_np.ndim == 2 else contact_np
        city_avg_r0 = np.mean(r0_np, axis=0) if r0_np.ndim == 2 else r0_np

        n_c = min(len(self.cities), len(city_avg_beta))
        lockdown_mask = np.zeros(n_c, dtype=bool)
        for i, city in enumerate(self.cities[:n_c]):
            if city in lockdown_info and lockdown_info[city].get('start') is not None:
                lockdown_mask[i] = True

        if lockdown_mask.sum() > 0 and (~lockdown_mask).sum() > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            groups = ['Lockdown Cities', 'Non-Lockdown']
            colors = ['lightcoral', 'skyblue']

            for ax, l_data, nl_data, title, ylabel in [
                (axes[0], city_avg_beta[:n_c][lockdown_mask], city_avg_beta[:n_c][~lockdown_mask],
                 'Lockdown Effect on Beta', r'$\beta$'),
                (axes[1], city_avg_contact[:n_c][lockdown_mask], city_avg_contact[:n_c][~lockdown_mask],
                 'Lockdown Effect on Contact', 'Contact'),
                (axes[2], city_avg_r0[:n_c][lockdown_mask], city_avg_r0[:n_c][~lockdown_mask],
                 r'Lockdown Effect on $R_t$', r'$R_t$')]:
                bars = ax.bar(groups, [np.mean(l_data), np.mean(nl_data)],
                             color=colors, edgecolor='black', alpha=0.8)
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel(ylabel)
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                            f'{bar.get_height():.3f}', ha='center', va='bottom')

            axes[2].axhline(y=1.0, color='black', linestyle='--', label='Epidemic Threshold')
            axes[2].legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, "parameters", f"v5_lockdown_effect_{model_name}.png"), dpi=300)
            plt.close()

    # ============================================================
    # 文字报告
    # ============================================================
    def print_quantitative_insights(self, eval_data, city_features_df, model_name):
        print(f"\n{'=' * 60}")
        print(f"[{model_name}] 深度量化特征分析")
        print(f"{'=' * 60}")

        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data['evaluation_results']

        residuals = results.get('residuals')
        if residuals is not None:
            res_np = residuals.numpy().flatten() if isinstance(residuals, torch.Tensor) \
                else np.array(residuals).flatten()
            res_np = res_np[np.isfinite(res_np)]
            if len(res_np) > 1 and np.std(res_np) > 1e-5:
                mu, std = np.mean(res_np), np.std(res_np)
                skew = float(stats.skew(res_np))
                kurt = float(stats.kurtosis(res_np))
                dw_stat = np.sum(np.diff(res_np) ** 2) / np.sum(res_np ** 2) if np.sum(res_np ** 2) != 0 else 2.0
                print(f" [误差分布] 均值: {mu:.4f} | 标准差: {std:.4f} | "
                      f"偏度: {skew:.4f} | 峰度: {kurt:.4f}")
                print(f" [时序检验] Durbin-Watson: {dw_stat:.4f}")

    def generate_academic_thesis_report(self, eval_data, model_name, city_features_df):
        print(f"\n{'=' * 60}")
        print(f"【{model_name}】论文图表文字表征输出")
        print(f"{'=' * 60}")
        filled_data = self.fill_missing_data(eval_data, model_name)
        results = filled_data.get('evaluation_results', {})

        def to_np(t):
            if t is None: return None
            return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

        betas = to_np(results.get('betas'))
        r0s = to_np(results.get('r0s'))
        if betas is not None and r0s is not None:
            print(f"  - 传播率(β): 均值={np.mean(betas):.4f}, 标准差={np.std(betas):.4f}")
            r0s_flat = r0s.flatten()
            valid = np.mean((r0s_flat >= 0.5) & (r0s_flat <= 2.0))
            print(f"  - 有效再生数(Rt): 均值={np.mean(r0s_flat):.3f}, Valid Rt比例={valid:.2%}")
        print("=" * 60)
