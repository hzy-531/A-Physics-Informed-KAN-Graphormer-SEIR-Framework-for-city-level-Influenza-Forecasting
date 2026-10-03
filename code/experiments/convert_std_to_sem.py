#!/usr/bin/env python
"""
把所有 mean ± std 统一换算成 mean ± SEM (standard error of the mean)。

换算公式:  SEM = std / sqrt(n),  n = 种子数 = 20,  即 std / 4.4721。

覆盖论文三张带 ± 的表:
  - tab:results    R² / RMSE / MAE  (Full+LSTM 来自 6models_200323; NoPhys+M 来自 align_003254)
  - tab:epi        β / γ⁻¹ / C / Rt  (Full 来自 200323; M 来自 align_003254)
  - tab:spatial    分层 pooled R²    (All / Tier-1 / Tier-2)

输出三列: mean / std(ddof=1) / SEM = std/sqrt(20)。
用法:  PYTHONIOENCODING=utf-8 python experiments/convert_std_to_sem.py
"""
import sys, os, pickle, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import __main__
from exp_lib import models, layers, physics, loss, data

_COMPAT_CLASSES = [
    ('EnhancedFull_Graphormer_V3', models.EnhancedFull_Graphormer_V3),
    ('EnhancedMLP_Baseline_V3', models.EnhancedMLP_Baseline_V3),
    ('EnhancedLSTM_Baseline_V3', models.EnhancedLSTM_Baseline_V3),
    ('EnhancedGAT_Baseline_V3', models.EnhancedGAT_Baseline_V3),
    ('EnhancedKAN_Only_V3', models.EnhancedKAN_Only_V3),
    ('M_Graphormer_Baseline', models.M_Graphormer_Baseline),
    ('NoPhysics_KAN_Graphormer', models.NoPhysics_KAN_Graphormer),
    ('RobustKANLinear', layers.RobustKANLinear),
    ('StandardMLP', layers.StandardMLP),
    ('GatedTCN', layers.GatedTCN),
    ('RBFSpatialEncoding', layers.RBFSpatialEncoding),
    ('GraphormerLayer', layers.GraphormerLayer),
    ('ImprovedGATLayer', layers.ImprovedGATLayer),
    ('DataCollector', data.DataCollector),
    ('CityLevelLogMinMaxScaler', data.CityLevelLogMinMaxScaler),
    ('CityLevelLinearMinMaxScaler', data.CityLevelLinearMinMaxScaler),
    ('TemporalFeatureEnhancer', data.TemporalFeatureEnhancer),
    ('ImprovedTemporalDataset', data.ImprovedTemporalDataset),
    ('CurriculumScientificLoss', loss.CurriculumScientificLoss),
]
for _name, _cls in _COMPAT_CLASSES:
    setattr(__main__, _name, _cls)

from sklearn.metrics import r2_score

FULL_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
MGRAPH_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_align_20260919_003254'
N = 20
SQRT_N = np.sqrt(N)

CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市']
TIER1 = ['北京市', '上海市', '广州市', '深圳市']
TIER2 = ['天津市', '重庆市', '成都市', '武汉市', '南京市', '杭州市', '西安市', '郑州市']


def to_np(t):
    if t is None:
        return None
    if isinstance(t, list):
        if len(t) == 0:
            return None
        t = torch.cat([x.cpu() if hasattr(x, 'cpu') else torch.tensor(x) for x in t])
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    elif hasattr(t, 'numpy'):
        t = t.numpy()
    return np.array(t)


def read_summary(csv_path):
    """读取 summary_20seeds.csv -> {model: {metric: (mean, std)}}"""
    out = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            m = row['model']
            out[m] = {
                'r2': (float(row['r2_mean']), float(row['r2_std'])),
                'rmse': (float(row['rmse_mean']), float(row['rmse_std'])),
                'mae': (float(row['mae_mean']), float(row['mae_std'])),
            }
    return out


def load_eval(result_dir, model):
    """读取 20 种子 evaluation_results, 返回 betas/gammas/contacts/r0s/predictions/targets."""
    keys = ['predictions', 'targets', 'betas', 'gammas', 'contacts', 'r0s']
    acc = {k: [] for k in keys}
    for s in range(N):
        pkl = os.path.join(result_dir, f'seed_{s}', 'results', f'{model}_result.pkl')
        with open(pkl, 'rb') as f:
            ed = pickle.load(f)
        er = ed['eval_data']['evaluation_results']
        for k in keys:
            acc[k].append(to_np(er[k]))
    return {k: np.stack(v, axis=0) for k, v in acc.items()}


def fmt(mean, std, dec_mean, dec_sem):
    sem = std / SQRT_N
    return mean, std, sem


def main():
    print("=" * 78)
    print(f"std -> SEM 换算  (SEM = std / sqrt({N}) = std / {SQRT_N:.4f})")
    print("=" * 78)

    # ---------- tab:results ----------
    print("\n【tab:results】R² / RMSE / MAE  (std -> SEM)")
    print(f"{'model':24s} {'metric':6s} {'mean':>10s} {'std':>10s} {'SEM':>10s}")
    print("-" * 78)
    s6 = read_summary(os.path.join(FULL_DIR, 'summary_20seeds.csv'))
    sal = read_summary(os.path.join(MGRAPH_DIR, 'summary_20seeds.csv'))
    # Full + LSTM 来自 200323; NoPhys + M 来自 align
    tab_res = [
        ('Full_KAN_Graphormer', s6['Full_KAN_Graphormer']),
        ('NoPhysics_KAN_Graphormer', sal['NoPhysics_KAN_Graphormer']),
        ('M_Graphormer_Baseline', sal['M_Graphormer_Baseline']),
        ('LSTM_Baseline', s6['LSTM_Baseline']),
    ]
    for model, d in tab_res:
        for metric in ['r2', 'rmse', 'mae']:
            mean, std = d[metric]
            print(f"{model:24s} {metric:6s} {mean:10.4f} {std:10.4f} {std/SQRT_N:10.4f}")

    # ---------- tab:epi ----------
    print("\n【tab:epi】β / γ⁻¹ / C / Rt  (std -> SEM)")
    print(f"{'model':24s} {'param':10s} {'mean':>10s} {'std':>10s} {'SEM':>10s}")
    print("-" * 78)
    full = load_eval(FULL_DIR, 'Full_KAN_Graphormer')
    mgraph = load_eval(MGRAPH_DIR, 'M_Graphormer_Baseline')
    for name, d in [('Full_KAN_Graphormer', full), ('M_Graphormer_Baseline', mgraph)]:
        beta_seed = d['betas'].mean(axis=(1, 2))
        gamma_inv_seed = (1.0 / d['gammas']).mean(axis=(1, 2))
        c_seed = d['contacts'].mean(axis=(1, 2))
        r0_seed = d['r0s'].mean(axis=(1, 2))
        for pname, arr in [('beta', beta_seed), ('gamma_inv', gamma_inv_seed),
                           ('C', c_seed), ('Rt', r0_seed)]:
            mean, std = arr.mean(), arr.std(ddof=1)
            print(f"{name:24s} {pname:10s} {mean:10.4f} {std:10.4f} {std/SQRT_N:10.4f}")

    # ---------- tab:spatial ----------
    print("\n【tab:spatial】分层 pooled R² (Full KAN-G, 200323)  (std -> SEM)")
    print(f"{'subset':14s} {'mean':>10s} {'std':>10s} {'SEM':>10s}")
    print("-" * 78)
    idx = {c: i for i, c in enumerate(CITIES)}
    groups = {
        'All': list(range(len(CITIES))),
        'Tier-1': [idx[c] for c in TIER1],
        'Tier-2': [idx[c] for c in TIER2],
    }
    for sub, cols in groups.items():
        vals = []
        for s in range(N):
            pred = full['predictions'][s][:, cols].ravel()
            targ = full['targets'][s][:, cols].ravel()
            vals.append(float(r2_score(targ, pred)))
        vals = np.array(vals)
        print(f"{sub:14s} {vals.mean():10.4f} {vals.std(ddof=1):10.4f} {vals.std(ddof=1)/SQRT_N:10.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
