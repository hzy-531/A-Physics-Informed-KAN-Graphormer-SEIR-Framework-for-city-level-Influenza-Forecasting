"""
数据模块 — 从 exp_chronological.py 提取
包含数据收集器、归一化器、时序增强、数据集类、数据加载函数
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import time
import pickle
from scipy.interpolate import interp1d
import warnings

from .config import (BASE_DIR, FILE_FLU_TARGET, FILE_CITY_FEAT, FILE_MIG_PATH,
                     FILE_DATASET_2020, TARGET_CITIES, TIER_1_CITIES, TIER_2_CITIES,
                     LOCKDOWN_INFO)

warnings.filterwarnings('ignore')


# ==========================================
# DataCollector — 数据收集管理器
# ==========================================

class DataCollector:
    """数据收集管理器 - 收集模型运行过程中的实际数据"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置收集器"""
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'epoch_times': [], 'learning_rates': []
        }
        self.evaluation_results = {
            'predictions': [], 'targets': [], 'betas': [], 'gammas': [],
            'contacts': [], 'r0s': [], 'attention_maps': [], 'residuals': []
        }
        self.model_parameters = {
            'weights': {}, 'biases': {}, 'kan_weights': {}
        }
        self.timestamps = {
            'training_start': None, 'training_end': None,
            'evaluation_start': None, 'evaluation_end': None
        }
        self.performance_metrics = {
            'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'mse': 0.0
        }

    def record_training_epoch(self, epoch, train_loss, val_loss, epoch_time, lr):
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['epoch_times'].append(epoch_time)
        self.training_history['learning_rates'].append(lr)

    def record_evaluation_batch(self, predictions, targets, betas=None, gammas=None,
                                 contacts=None, r0s=None, attention_map=None):
        # 提取预测窗口的第一天(0)，并沿年龄组求和(dim=-1)
        if predictions.ndim == 4:
            pred_to_store = predictions[:, 0, :, :].sum(dim=-1)
            target_to_store = targets[:, 0, :, :].sum(dim=-1)
        elif predictions.ndim == 3:
            pred_to_store = predictions[:, 0, :].sum(dim=-1)
            target_to_store = targets[:, 0, :].sum(dim=-1)
        else:
            pred_to_store = predictions
            target_to_store = targets

        self.evaluation_results['predictions'].append(pred_to_store.detach().cpu())
        self.evaluation_results['targets'].append(target_to_store.detach().cpu())

        if betas is not None:
            self.evaluation_results['betas'].append(betas.detach().cpu())
        if gammas is not None:
            self.evaluation_results['gammas'].append(gammas.detach().cpu())
        if contacts is not None:
            self.evaluation_results['contacts'].append(contacts.detach().cpu())
        if r0s is not None:
            self.evaluation_results['r0s'].append(r0s.detach().cpu())
        if attention_map is not None:
            self.evaluation_results['attention_maps'].append(attention_map.detach().cpu())

        residuals = predictions - targets
        self.evaluation_results['residuals'].append(residuals.detach().cpu())

    def record_model_parameters(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    self.model_parameters['weights'][name] = param.data.clone().cpu()
                elif 'bias' in name:
                    self.model_parameters['biases'][name] = param.data.clone().cpu()
                elif 'spline' in name or 'kan' in name:
                    self.model_parameters['kan_weights'][name] = param.data.clone().cpu()

    def record_performance_metrics(self, r2, rmse, mae, mse):
        self.performance_metrics['r2'] = r2
        self.performance_metrics['rmse'] = rmse
        self.performance_metrics['mae'] = mae
        self.performance_metrics['mse'] = mse

    def start_training_timer(self):
        self.timestamps['training_start'] = time.time()

    def end_training_timer(self):
        self.timestamps['training_end'] = time.time()

    def start_evaluation_timer(self):
        self.timestamps['evaluation_start'] = time.time()

    def end_evaluation_timer(self):
        self.timestamps['evaluation_end'] = time.time()

    def get_training_time(self):
        if self.timestamps['training_start'] and self.timestamps['training_end']:
            return self.timestamps['training_end'] - self.timestamps['training_start']
        return 0.0

    def get_evaluation_time(self):
        if self.timestamps['evaluation_start'] and self.timestamps['evaluation_end']:
            return self.timestamps['evaluation_end'] - self.timestamps['evaluation_start']
        return 0.0

    def get_consolidated_data(self):
        consolidated = {}
        consolidated['training_history'] = self.training_history.copy()
        consolidated['evaluation_results'] = {}
        for key in self.evaluation_results:
            if self.evaluation_results[key]:
                if isinstance(self.evaluation_results[key][0], torch.Tensor):
                    consolidated['evaluation_results'][key] = torch.cat(
                        self.evaluation_results[key], dim=0)
                else:
                    consolidated['evaluation_results'][key] = self.evaluation_results[key]
            else:
                consolidated['evaluation_results'][key] = None
        consolidated['model_parameters'] = self.model_parameters.copy()
        consolidated['performance_metrics'] = self.performance_metrics.copy()
        consolidated['timestamps'] = self.timestamps.copy()
        consolidated['training_time'] = self.get_training_time()
        consolidated['evaluation_time'] = self.get_evaluation_time()
        return consolidated


# ==========================================
# 归一化器 (City-Level)
# ==========================================

class CityLevelLogMinMaxScaler:
    """逐城市对数极差归一化"""

    def __init__(self):
        self.min_vals = None
        self.range_vals = None
        self.data_min_ = None
        self.data_range_ = None

    def fit(self, X_3d):
        X_log = np.log1p(np.clip(X_3d, 0, None))
        self.min_vals = np.min(X_log, axis=0, keepdims=True)
        self.max_vals = np.max(X_log, axis=0, keepdims=True)
        self.range_vals = self.max_vals - self.min_vals

        static_mask = (self.range_vals < 1e-8)
        if np.any(static_mask):
            global_min = np.min(X_log, axis=(0, 1), keepdims=True)
            global_max = np.max(X_log, axis=(0, 1), keepdims=True)
            global_range = global_max - global_min
            global_range[global_range < 1e-8] = 1.0
            self.min_vals = np.where(static_mask, global_min, self.min_vals)
            self.range_vals = np.where(static_mask, global_range, self.range_vals)

        self.range_vals[self.range_vals < 1e-8] = 1.0
        self.data_min_ = torch.tensor(self.min_vals, dtype=torch.float32)
        self.data_range_ = torch.tensor(self.range_vals, dtype=torch.float32)
        return self

    def transform(self, X_3d):
        X_log = np.log1p(np.clip(X_3d, 0, None))
        return (X_log - self.min_vals) / self.range_vals

    def inverse_transform(self, X_scaled_3d):
        X_log = X_scaled_3d * self.range_vals + self.min_vals
        return np.expm1(X_log)


class CityLevelLinearMinMaxScaler:
    """逐城市线性极差归一化（随机划分专用）"""

    def __init__(self):
        self.min_vals = None
        self.range_vals = None
        self.data_min_ = None
        self.data_range_ = None

    def fit(self, X_3d):
        X_lin = np.clip(X_3d, 0, None)
        self.min_vals = np.min(X_lin, axis=0, keepdims=True)
        self.max_vals = np.max(X_lin, axis=0, keepdims=True)
        self.range_vals = self.max_vals - self.min_vals

        static_mask = (self.range_vals < 1e-8)
        if np.any(static_mask):
            global_min = np.min(X_lin, axis=(0, 1), keepdims=True)
            global_max = np.max(X_lin, axis=(0, 1), keepdims=True)
            global_range = global_max - global_min
            global_range[global_range < 1e-8] = 1.0
            self.min_vals = np.where(static_mask, global_min, self.min_vals)
            self.range_vals = np.where(static_mask, global_range, self.range_vals)

        self.range_vals[self.range_vals < 1e-8] = 1.0
        self.data_min_ = torch.tensor(self.min_vals, dtype=torch.float32)
        self.data_range_ = torch.tensor(self.range_vals, dtype=torch.float32)
        return self

    def transform(self, X_3d):
        X_lin = np.clip(X_3d, 0, None)
        return (X_lin - self.min_vals) / self.range_vals

    def inverse_transform(self, X_scaled_3d):
        X_lin = X_scaled_3d * self.range_vals + self.min_vals
        return X_lin


# ==========================================
# TemporalFeatureEnhancer — 时序特征增强器
# ==========================================

class TemporalFeatureEnhancer:
    """时序特征增强器——将11维特征扩展为17维（与主线一致）"""

    @staticmethod
    def add_temporal_features(features, dates):
        n_days, n_cities, n_features = features.shape
        case_growth = np.zeros((n_days, n_cities, 1))
        if n_days > 1:
            current = features[1:, :, 0]
            prev = features[:-1, :, 0]
            safe_prev = np.where(prev == 0, 1e-6, prev)
            growth = current / safe_prev - 1.0
            case_growth[1:, :, 0] = np.clip(growth, -0.8, 2.0)

        window = min(7, n_days)
        moving_avg = np.zeros((n_days, n_cities, 1))
        moving_std = np.zeros((n_days, n_cities, 1))
        for i in range(n_days):
            start = max(0, i - window + 1)
            if start <= i:
                win = features[start:i + 1, :, 0]
                if len(win) > 0:
                    moving_avg[i, :, 0] = win.mean(axis=0)
                if len(win) > 1:
                    moving_std[i, :, 0] = win.std(axis=0)

        safe_avg = np.where(moving_avg[:, :, 0] == 0, 1e-6, moving_avg[:, :, 0])
        case_ratio = np.zeros((n_days, n_cities, 1))
        case_ratio[:, :, 0] = np.clip(features[:, :, 0] / safe_avg, 0.0, 3.0)

        weekday_factor = np.zeros((n_days, n_cities, 1))
        for i, d in enumerate(dates):
            try:
                wd = pd.Timestamp(d).weekday() if isinstance(d, str) else d.weekday()
                weekday_factor[i, :, 0] = 0.8 + 0.1 * (wd / 6)
            except:
                weekday_factor[i, :, 0] = 0.85

        phase = np.zeros((n_days, n_cities, 1))
        for i in range(n_days):
            if i < min(20, n_days):
                phase[i, :, 0] = 0.0
            elif i < min(40, n_days):
                phase[i, :, 0] = 0.5
            else:
                phase[i, :, 0] = 1.0

        temporal = np.concatenate([
            case_growth, moving_avg, case_ratio, moving_std, weekday_factor, phase
        ], axis=-1)
        return np.concatenate([features, temporal], axis=-1)


# ==========================================
# ImprovedTemporalDataset — 时序数据集
# ==========================================

class ImprovedTemporalDataset(Dataset):
    def __init__(self, features, adjs, pops, lookback=7, predict=3,
                 augment_prob=0.0, noise_level=0.03, phase_stratified=True,
                 variance_threshold=0.005, indices=None,
                 case_scaler=None, other_scaler=None):
        self.case_scaler = case_scaler
        self.other_scaler = other_scaler
        self.features_raw = torch.tensor(features, dtype=torch.float32)
        self.features = self._normalize_features()
        self.adjs = adjs
        self.pops = pops
        self.lookback = lookback
        self.predict = predict
        self.augment_prob = augment_prob
        self.noise_level = noise_level
        self.phase_stratified = phase_stratified
        self.variance_threshold = variance_threshold
        self._training = True

        self.total_len = len(features) - lookback - predict + 1
        if self.total_len <= 0:
            raise ValueError(f"数据长度不足")

        if indices is None:
            if phase_stratified:
                self.indices = self._create_stratified_indices()
            else:
                self.indices = list(range(self.total_len))
        else:
            self.indices = indices

        print(f"    数据集创建完成: {len(self.indices)} 个样本")
        self._check_variance()

    def _normalize_features(self):
        feat_np = self.features_raw.clone().numpy()
        if self.case_scaler is not None:
            feat_np[:, :, :3] = self.case_scaler.transform(feat_np[:, :, :3])
        if self.other_scaler is not None:
            feat_np[:, :, 3:] = self.other_scaler.transform(feat_np[:, :, 3:])
        return torch.tensor(feat_np, dtype=torch.float32)

    def _create_stratified_indices(self):
        n_days = len(self.features_raw)
        phase_indices = {'rise': [], 'stable': [], 'decline': []}
        for idx in range(self.total_len):
            start_day = idx + self.lookback
            if start_day < 20:
                phase = 'rise'
            elif start_day < 40:
                phase = 'stable'
            else:
                phase = 'decline'
            phase_indices[phase].append(idx)
        phase_weights = {'rise': 0.30, 'stable': 0.30, 'decline': 0.40}
        total_samples = min(self.total_len, 250)
        stratified_indices = []
        for phase, weight in phase_weights.items():
            phase_samples = int(total_samples * weight)
            available = len(phase_indices[phase])
            if available == 0:
                continue
            phase_samples = min(phase_samples, available)
            selected = np.random.choice(phase_indices[phase], phase_samples,
                                        replace=False)
            stratified_indices.extend(selected.tolist())
        if len(stratified_indices) < 30:
            stratified_indices = list(range(self.total_len))
        np.random.shuffle(stratified_indices)
        print(f"    分层采样: 上升期{len(phase_indices['rise'])}个, "
              f"平稳期{len(phase_indices['stable'])}个, "
              f"下降期{len(phase_indices['decline'])}个 -> "
              f"采样后共{len(stratified_indices)}个")
        return stratified_indices

    def _check_variance(self):
        all_feat = self.features.reshape(-1, self.features.shape[-1])
        var_per_feat = all_feat.var(dim=0)
        if var_per_feat.mean() < self.variance_threshold:
            noise = torch.randn_like(self.features) * self.variance_threshold * 2
            self.features = self.features + noise
            print(f"    ⚠️ 特征方差过低({var_per_feat.mean():.6f})，已添加增强噪声")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x = self.features[original_idx: original_idx + self.lookback]
        y_norm = self.features[original_idx + self.lookback:
                               original_idx + self.lookback + self.predict, :, 0:3]
        y_raw = self.features_raw[original_idx + self.lookback:
                                  original_idx + self.lookback + self.predict, :, 0:3]
        adj_seq = self.adjs[original_idx + self.lookback:
                            original_idx + self.lookback + self.predict]

        if self.training and torch.rand(1).item() < self.augment_prob:
            x = self._apply_augmentation(x)

        return x, adj_seq, y_norm, y_raw

    def _apply_augmentation(self, x):
        device = x.device
        if torch.rand(1).item() < 0.4:
            shift = torch.randint(-2, 3, (1,)).item()
            if shift != 0 and abs(shift) < x.shape[0]:
                x = torch.roll(x, shift, dims=0)
                if shift > 0:
                    x[:shift] = x[shift:2 * shift].flip(0)
                else:
                    x[shift:] = x[2 * shift:shift].flip(0)
        if torch.rand(1).item() < 0.7:
            noise_scale = self.noise_level * 0.8
            noise_x = torch.randn_like(x) * noise_scale
            time_factor = torch.sin(
                torch.arange(x.shape[0], device=device).float().view(-1, 1, 1) * 0.3
            )
            noise_x = noise_x + time_factor * self.noise_level * 0.2
            x = x + noise_x
        x = torch.clamp(x, -1.5, 3.0)
        return x

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value


# ==========================================
# 数据预处理辅助函数
# ==========================================

def preprocess_intra_city_data(df):
    """处理城内出行数据的缺失值"""
    print(">>> [Data Preprocess] 处理城内出行数据缺失值...")
    cols_to_fix = ['公交百度搜索指数', '地铁百度搜索指数']
    for col in cols_to_fix:
        if col not in df.columns:
            continue
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby('City_Name')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        df[col] = df[col].fillna(df[col].median() if df[col].median() > 0 else 0)
    return df


def get_smooth_age_ratios(dates):
    """使用三次样条插值生成逐日平滑的年龄比例"""
    anchors = {
        1: [0.40, 0.50, 0.10],
        2: [0.25, 0.65, 0.10],
        3: [0.30, 0.60, 0.10]
    }
    ts_anchors = []
    val_anchors = []
    for m in [1, 2, 3]:
        t = pd.Timestamp(f"2020-{m:02d}-15").timestamp()
        ts_anchors.append(t)
        val_anchors.append(anchors[m])

    ts_anchors = [pd.Timestamp("2020-01-01").timestamp()] + ts_anchors + \
                 [pd.Timestamp("2020-04-01").timestamp()]
    val_anchors = [anchors[1]] + val_anchors + [anchors[3]]
    val_anchors = np.array(val_anchors)

    f_child = interp1d(ts_anchors, val_anchors[:, 0], kind='cubic',
                       fill_value="extrapolate")
    f_adult = interp1d(ts_anchors, val_anchors[:, 1], kind='cubic',
                       fill_value="extrapolate")
    f_elder = interp1d(ts_anchors, val_anchors[:, 2], kind='cubic',
                       fill_value="extrapolate")

    daily_ratios = {}
    for date in dates:
        ts = pd.Timestamp(date).timestamp()
        r = np.array([f_child(ts), f_adult(ts), f_elder(ts)])
        daily_ratios[date] = np.clip(r, 0, 1)
        daily_ratios[date] = daily_ratios[date] / np.sum(daily_ratios[date])
    return daily_ratios


def compute_lockdown_factor(city, date, lockdown_info=None):
    """计算封城干预强度因子"""
    if lockdown_info is None:
        lockdown_info = LOCKDOWN_INFO
    if city not in lockdown_info or lockdown_info[city]['start'] is None:
        return 0.0

    lockdown_start = pd.Timestamp(lockdown_info[city]['start'])
    lockdown_end = pd.Timestamp(lockdown_info[city]['end']) if lockdown_info[city]['end'] \
        else pd.Timestamp('2020-04-30')
    strength = lockdown_info[city]['strength']
    current_date = pd.Timestamp(date)

    if current_date < lockdown_start:
        return 0.0
    elif lockdown_start <= current_date <= lockdown_end:
        days_in_lockdown = (current_date - lockdown_start).days
        max_days = (lockdown_end - lockdown_start).days
        if max_days > 0:
            progression = min(1.0, days_in_lockdown / 14)
            return strength * progression
        return strength
    else:
        days_after = (current_date - lockdown_end).days
        recovery = max(0.0, 1.0 - days_after / 21)
        return strength * recovery


def add_city_specific_features(df_static, city):
    """添加核心城市特异性特征，包括区域标识"""
    features = {}
    row = df_static[df_static['城市名称'] == city]
    if len(row) == 0:
        features['population_density'] = 1000.0
        features['city_tier'] = 1.0 if city in TIER_1_CITIES else \
            (2.0 if city in TIER_2_CITIES else 3.0)
        features['heating'] = 1.0 if city in [
            '北京市', '天津市', '石家庄市', '沈阳市', '哈尔滨市', '长春市'] else 0.0
        features['region'] = 0
    else:
        population = row['2019市常住人口(万人)'].values[0]
        area = 5000.0
        features['population_density'] = population * 10000 / area if area > 0 else 1000.0
        features['city_tier'] = 1.0 if city in TIER_1_CITIES else \
            (2.0 if city in TIER_2_CITIES else 3.0)
        features['heating'] = 1.0 if row['区域标识 (Region)(南方=1，北方=0)'].values[0] == 0 else 0.0
        features['region'] = row['区域标识 (Region)(南方=1，北方=0)'].values[0]
    return features


# ==========================================
# 原始数据加载
# ==========================================

def load_raw_data_only():
    """只加载原始数据，不进行任何归一化（与主线特征完全一致）"""
    print(">>> [Data Load] 加载原始数据...")
    df_flu = pd.read_excel(FILE_FLU_TARGET)
    df_flu['日期 (Date)'] = pd.to_datetime(df_flu['日期 (Date)'])
    df_flu['病例_原始'] = df_flu['城市_日估算病例 (City_Daily_Case)'].clip(lower=0)

    if os.path.exists(FILE_DATASET_2020):
        df_env = pd.read_excel(FILE_DATASET_2020)
        df_env.rename(columns={'日期': 'Date', '城市': 'City_Name'}, inplace=True)
        df_env['Date'] = pd.to_datetime(df_env['Date'])
        df_env = preprocess_intra_city_data(df_env)
    else:
        df_env = pd.DataFrame()

    df_static = pd.read_excel(FILE_CITY_FEAT)

    cities = [c for c in TARGET_CITIES if c in df_flu['城市名称 (City_Name)'].unique()]
    if len(cities) < len(TARGET_CITIES):
        cities = TARGET_CITIES

    dates = sorted(df_flu['日期 (Date)'].unique())
    age_ratios = get_smooth_age_ratios(dates)

    FEATURE_DIM_BASE = 11
    features_raw_list = []
    for date in dates:
        ratio = age_ratios[date]
        day_feat = np.zeros((len(cities), FEATURE_DIM_BASE))
        current_flu = df_flu[df_flu['日期 (Date)'] == date]
        current_env = df_env[df_env['Date'] == date] if not df_env.empty else pd.DataFrame()
        for city_idx, city in enumerate(cities):
            city_features = add_city_specific_features(df_static, city)

            flu_row = current_flu[current_flu['城市名称 (City_Name)'] == city]
            if len(flu_row) > 0:
                total = flu_row['病例_原始'].values[0]
                day_feat[city_idx, 0] = total * ratio[0]
                day_feat[city_idx, 1] = total * ratio[1]
                day_feat[city_idx, 2] = total * ratio[2]
            else:
                day_feat[city_idx, 0:3] = 0

            if not current_env.empty:
                env_row = current_env[current_env['City_Name'] == city]
                if len(env_row) > 0:
                    bus = env_row['公交百度搜索指数'].values[0] if '公交百度搜索指数' in env_row.columns else 0
                    subway = env_row['地铁百度搜索指数'].values[0] if '地铁百度搜索指数' in env_row.columns else 0
                    temp = env_row['最终气温(℃)'].values[0] if '最终气温(℃)' in env_row.columns else 10
                    day_feat[city_idx, 3] = bus
                    day_feat[city_idx, 4] = subway
                    day_feat[city_idx, 5] = temp
                else:
                    day_feat[city_idx, 3] = 0
                    day_feat[city_idx, 4] = 0
                    day_feat[city_idx, 5] = 10
            else:
                day_feat[city_idx, 3] = 0
                day_feat[city_idx, 4] = 0
                day_feat[city_idx, 5] = 10

            day_feat[city_idx, 6] = city_features['population_density']
            day_feat[city_idx, 7] = city_features['city_tier']
            day_feat[city_idx, 8] = city_features['heating']
            day_feat[city_idx, 9] = compute_lockdown_factor(city, date, LOCKDOWN_INFO)
            day_feat[city_idx, 10] = city_features['region']
        features_raw_list.append(day_feat)

    features_raw = np.array(features_raw_list)
    features_enhanced = TemporalFeatureEnhancer.add_temporal_features(features_raw, dates)

    # 迁徙矩阵
    xls = pd.ExcelFile(FILE_MIG_PATH)
    sheet_names = [name for name in xls.sheet_names if name.startswith('D')]
    adj_list = []
    for date in dates:
        d_str = pd.to_datetime(date).strftime("D%Y%m%d")
        if d_str in sheet_names:
            df_mat = pd.read_excel(xls, sheet_name=d_str, index_col=0)
            df_mat = df_mat.reindex(index=cities, columns=cities, fill_value=0.0)
            row_sums = df_mat.sum(axis=1)
            row_sums[row_sums == 0] = 1
            df_mat = df_mat.div(row_sums, axis=0)
            adj_list.append(df_mat.values)
        else:
            adj_list.append(adj_list[-1].copy() if len(adj_list) > 0
                            else np.eye(len(cities)))
    adjs = np.array(adj_list)

    # 人口数据
    pops = torch.zeros(len(cities), 3)
    for i, city in enumerate(cities):
        row = df_static[df_static['城市名称'] == city]
        if len(row) > 0:
            total = row['2019市常住人口(万人)'].values[0] * 10000
            pops[i, 0] = total * 0.15
            pops[i, 1] = total * 0.70
            pops[i, 2] = total * 0.15
        else:
            pops[i, :] = torch.tensor([1e6, 5e6, 1e6])

    return features_enhanced, adjs, pops, cities, dates


def build_enhanced_dataset_with_scalers(train_indices, lookback=7, predict=3):
    """
    构建数据集，并基于训练集索引拟合归一化器
    返回: features_raw, adjs, pops, case_scaler, other_scaler, cities, dates
    """
    print(">>> [Step 1] 构建原始多源融合数据集（未归一化）...")
    df_flu = pd.read_excel(FILE_FLU_TARGET)
    df_flu['日期 (Date)'] = pd.to_datetime(df_flu['日期 (Date)'])
    df_flu['病例_原始'] = df_flu['城市_日估算病例 (City_Daily_Case)'].clip(lower=0)

    if os.path.exists(FILE_DATASET_2020):
        df_env = pd.read_excel(FILE_DATASET_2020)
        df_env.rename(columns={'日期': 'Date', '城市': 'City_Name'}, inplace=True)
        df_env['Date'] = pd.to_datetime(df_env['Date'])
        df_env = preprocess_intra_city_data(df_env)
    else:
        df_env = pd.DataFrame()

    df_static = pd.read_excel(FILE_CITY_FEAT)

    cities = [c for c in TARGET_CITIES if c in df_flu['城市名称 (City_Name)'].unique()]
    if len(cities) < len(TARGET_CITIES):
        print(f"    警告：仅找到 {len(cities)} 个目标城市，使用全部目标城市")
        cities = TARGET_CITIES

    dates = sorted(df_flu['日期 (Date)'].unique())
    print(f"    日期范围: {dates[0]} 到 {dates[-1]}, 共 {len(dates)} 天")

    age_ratios = get_smooth_age_ratios(dates)

    FEATURE_DIM_BASE = 10
    features_raw_list = []
    for date in dates:
        ratio = age_ratios[date]
        day_feat = np.zeros((len(cities), FEATURE_DIM_BASE))
        current_flu = df_flu[df_flu['日期 (Date)'] == date]
        current_env = df_env[df_env['Date'] == date] if not df_env.empty else pd.DataFrame()
        for city_idx, city in enumerate(cities):
            city_features = add_city_specific_features(df_static, city)

            flu_row = current_flu[current_flu['城市名称 (City_Name)'] == city]
            if len(flu_row) > 0:
                total_cases = flu_row['病例_原始'].values[0]
                day_feat[city_idx, 0] = total_cases * ratio[0]
                day_feat[city_idx, 1] = total_cases * ratio[1]
                day_feat[city_idx, 2] = total_cases * ratio[2]
            else:
                day_feat[city_idx, 0] = 0
                day_feat[city_idx, 1] = 0
                day_feat[city_idx, 2] = 0

            if not current_env.empty:
                env_row = current_env[current_env['City_Name'] == city]
                if len(env_row) > 0:
                    bus = env_row['公交百度搜索指数'].values[0] if '公交百度搜索指数' in env_row.columns else 0
                    subway = env_row['地铁百度搜索指数'].values[0] if '地铁百度搜索指数' in env_row.columns else 0
                    temp = env_row['最终气温(℃)'].values[0] if '最终气温(℃)' in env_row.columns else 10
                    day_feat[city_idx, 3] = bus
                    day_feat[city_idx, 4] = subway
                    day_feat[city_idx, 5] = temp
                else:
                    day_feat[city_idx, 3] = 0
                    day_feat[city_idx, 4] = 0
                    day_feat[city_idx, 5] = 10
            else:
                day_feat[city_idx, 3] = 0
                day_feat[city_idx, 4] = 0
                day_feat[city_idx, 5] = 10

            city_features = add_city_specific_features(df_static, city)
            day_feat[city_idx, 6] = city_features['population_density']
            day_feat[city_idx, 7] = city_features['city_tier']
            day_feat[city_idx, 8] = city_features['heating']
            day_feat[city_idx, 9] = compute_lockdown_factor(city, date, LOCKDOWN_INFO)
        features_raw_list.append(day_feat)

    features_raw = np.array(features_raw_list)
    features_enhanced = TemporalFeatureEnhancer.add_temporal_features(features_raw, dates)
    print(f"    增强后特征维度: {features_enhanced.shape[-1]}")

    # 迁徙矩阵
    xls = pd.ExcelFile(FILE_MIG_PATH)
    sheet_names = [name for name in xls.sheet_names if name.startswith('D')]
    adj_list = []
    for date in dates:
        d_str = pd.to_datetime(date).strftime("D%Y%m%d")
        if d_str in sheet_names:
            df_mat = pd.read_excel(xls, sheet_name=d_str, index_col=0)
            df_mat = df_mat.reindex(index=cities, columns=cities, fill_value=0.0)
            row_sums = df_mat.sum(axis=1)
            row_sums[row_sums == 0] = 1
            df_mat = df_mat.div(row_sums, axis=0)
            adj_list.append(df_mat.values)
        else:
            adj_list.append(adj_list[-1].copy() if len(adj_list) > 0
                            else np.eye(len(cities)))
    adjs = np.array(adj_list)

    # 人口
    pops = torch.zeros(len(cities), 3)
    for i, city in enumerate(cities):
        row = df_static[df_static['城市名称'] == city]
        if len(row) > 0:
            total = row['2019市常住人口(万人)'].values[0] * 10000
            pops[i, 0] = total * 0.15
            pops[i, 1] = total * 0.70
            pops[i, 2] = total * 0.15
        else:
            pops[i, :] = torch.tensor([1e6, 5e6, 1e6])

    # 基于训练集拟合 scaler
    total_samples = len(features_enhanced) - lookback - predict + 1
    train_days_set = set()
    for idx in train_indices:
        for t in range(idx, idx + lookback + predict):
            train_days_set.add(t)
    train_days = sorted(train_days_set)
    train_features = features_enhanced[train_days]
    train_features_np = train_features.numpy() if isinstance(train_features, torch.Tensor) else train_features

    case_scaler = CityLevelLogMinMaxScaler()
    case_scaler.fit(train_features_np[:, :, :3])
    other_scaler = CityLevelLogMinMaxScaler()
    other_scaler.fit(train_features_np[:, :, 3:])

    print(f"    病例特征范围: min={case_scaler.data_min_}, max 已设置")
    print(f"    其他特征范围: min={other_scaler.data_min_}, max 已设置")

    return features_enhanced, adjs, pops, case_scaler, other_scaler, cities, dates
