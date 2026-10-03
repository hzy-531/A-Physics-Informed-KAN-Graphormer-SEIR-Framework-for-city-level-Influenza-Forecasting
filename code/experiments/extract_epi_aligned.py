#!/usr/bin/env python
"""
从 20 种子结果里提取 tab:epi 所需的流行病学参数 (β / γ⁻¹ / C / Rt)。

先对 original 的 Full KAN-G 与 M-Graphormer 跑一遍验证公式能复现论文现值,
再对 aligned 的 M-Graphormer 算出新值。

用法:
    PYTHONIOENCODING=utf-8 python experiments/extract_epi_aligned.py
"""
import os, sys, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_lib.data  # noqa: F401  (pkl 反序列化需要 exp_lib 可导入)
import exp_lib.trainer  # noqa: F401

ORIG_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_6models_20260915_200323'
ALIGN_DIR = r'E:\Claude code\KAN+\result\ablation_20seeds_align_20260919_003254'
N_SEEDS = 20


def to_np(t):
    if t is None:
        return None
    if isinstance(t, list):
        if len(t) == 0:
            return None
        t = torch.cat([x.cpu() if hasattr(x, 'cpu') else torch.tensor(x) for x in t])
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    elif isinstance(t, torch.Tensor):
        t = t.numpy()
    return np.array(t)


def extract(result_dir, model):
    betas, gammas, contacts, r0s = [], [], [], []
    for s in range(N_SEEDS):
        pkl = os.path.join(result_dir, f'seed_{s}', 'results', f'{model}_result.pkl')
        with open(pkl, 'rb') as f:
            ed = pickle.load(f)
        er = ed['eval_data']['evaluation_results']
        b = to_np(er.get('betas'))
        g = to_np(er.get('gammas'))
        c = to_np(er.get('contacts'))
        r = to_np(er.get('r0s'))
        betas.append(float(b.mean()))
        gammas.append(float(g.mean()))
        contacts.append(float(c.mean()))
        r0s.append(float(r.mean()))

    betas = np.array(betas)
    gammas = np.array(gammas)
    contacts = np.array(contacts)
    r0s = np.array(r0s)
    # γ⁻¹ 两种口径
    inv_mean_of_inv = np.array([float((1.0 / g).mean()) for g in gammas]) if False else None
    # 逐 cell 取 1/g 再 mean: 需要重算
    inv_cells = []
    for s in range(N_SEEDS):
        pkl = os.path.join(result_dir, f'seed_{s}', 'results', f'{model}_result.pkl')
        with open(pkl, 'rb') as f:
            ed = pickle.load(f)
        g = to_np(ed['eval_data']['evaluation_results'].get('gammas'))
        inv_cells.append(float(np.mean(1.0 / g)))
    inv_cells = np.array(inv_cells)

    def ms(v):
        return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"

    print(f"--- {model} ({os.path.basename(result_dir)}) ---")
    print(f"  β   = {ms(betas)}")
    print(f"  γ   = {ms(gammas)}   (1/mean γ = {1.0/gammas.mean():.4f})")
    print(f"  γ⁻¹ (逐cell mean(1/g)) = {ms(inv_cells)}")
    print(f"  C   = {ms(contacts)}")
    print(f"  Rt  = {ms(r0s)}")
    print()


if __name__ == '__main__':
    # 1) 验证: original Full KAN-G 应复现 0.638 / 4.20 / 0.788 / 1.969
    extract(ORIG_DIR, 'Full_KAN_Graphormer')
    # 2) 验证: original M-Graphormer 应复现 0.627 / 4.24 / 0.795 / 1.969
    extract(ORIG_DIR, 'M_Graphormer_Baseline')
    # 3) 新值: aligned M-Graphormer
    extract(ALIGN_DIR, 'M_Graphormer_Baseline')
