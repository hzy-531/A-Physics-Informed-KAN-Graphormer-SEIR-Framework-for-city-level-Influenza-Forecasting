"""
SEIR 物理动力学模块 — 从 exp_chronological.py 提取
多斑块 SEIR 系统的微分方程、RK4 求解器、下一代矩阵 R0 计算
"""
import torch
import numpy as np


def compute_derivatives_seir(S, E, I, R, beta, gamma, sigma_incubation, contact_mod,
                              migration_matrix, population, base_contact_matrix, mobility_rate):
    """【学术重构】：升级为纳入潜伏期的多斑块 SEIR 动力学内核"""
    eps = 1e-8
    mu = 1.0 / (70 * 365)
    lambda_dis = 0.00008

    S_safe = torch.clamp(S, min=0)
    E_safe = torch.clamp(E, min=0)
    I_safe = torch.clamp(I, min=0)

    i_frac = I_safe / (population + eps)
    contact_mod_exp = contact_mod.unsqueeze(-1).unsqueeze(-1)
    base = base_contact_matrix.float().unsqueeze(0).unsqueeze(0)
    contact_dynamic = base * contact_mod_exp

    risk = torch.einsum('bnij,bnj->bni', contact_dynamic, i_frac)
    lambda_t = beta.unsqueeze(-1) * risk

    infection_rate = lambda_t * S_safe
    incubation_rate = sigma_incubation * E_safe
    recovery_rate = gamma.unsqueeze(-1) * I_safe

    B_dim, N_dim, _ = migration_matrix.shape
    identity = torch.eye(N_dim, device=migration_matrix.device).unsqueeze(0)
    effective_mig = (1 - mobility_rate) * identity + mobility_rate * migration_matrix
    mig_T = effective_mig.transpose(1, 2)

    B_rate = mu * population
    dS_dt = B_rate - mu * S_safe - infection_rate + torch.bmm(mig_T, S_safe) - S_safe
    dE_dt = infection_rate - (mu + sigma_incubation) * E_safe + torch.bmm(mig_T, E_safe) - E_safe
    dI_dt = incubation_rate - (mu + lambda_dis) * I_safe - recovery_rate + torch.bmm(mig_T, I_safe) - I_safe
    dR_dt = recovery_rate - mu * R + torch.bmm(mig_T, R) - R

    return dS_dt, dE_dt, dI_dt, dR_dt, incubation_rate


def age_structured_seir_step(S_t, E_t, I_t, R_t, beta, gamma, contact_mod, migration_matrix,
                              population, base_contact_matrix, mobility_rate=0.01, dt=1.0):
    """【核心物理修复】：利用RK4求解SEIR微分方程，消除SIR模型的相位偏移误差"""
    S_t, E_t, I_t, R_t = S_t.float(), E_t.float(), I_t.float(), R_t.float()
    beta, gamma, contact_mod = beta.float(), gamma.float(), contact_mod.float()
    migration_matrix = migration_matrix.float()
    # 潜伏期常数，假设平均潜伏期为3天
    sigma_incubation = 1.0 / 3.0

    num_steps = 2
    sub_dt = dt / num_steps

    incidence_final = torch.zeros_like(I_t)
    S_curr, E_curr, I_curr, R_curr = S_t, E_t, I_t, R_t

    for _ in range(num_steps):
        k1_S, k1_E, k1_I, k1_R, inc1 = compute_derivatives_seir(
            S_curr, E_curr, I_curr, R_curr, beta, gamma, sigma_incubation,
            contact_mod, migration_matrix, population, base_contact_matrix, mobility_rate)
        k2_S, k2_E, k2_I, k2_R, inc2 = compute_derivatives_seir(
            S_curr + 0.5 * sub_dt * k1_S, E_curr + 0.5 * sub_dt * k1_E,
            I_curr + 0.5 * sub_dt * k1_I, R_curr + 0.5 * sub_dt * k1_R,
            beta, gamma, sigma_incubation, contact_mod, migration_matrix,
            population, base_contact_matrix, mobility_rate)
        k3_S, k3_E, k3_I, k3_R, inc3 = compute_derivatives_seir(
            S_curr + 0.5 * sub_dt * k2_S, E_curr + 0.5 * sub_dt * k2_E,
            I_curr + 0.5 * sub_dt * k2_I, R_curr + 0.5 * sub_dt * k2_R,
            beta, gamma, sigma_incubation, contact_mod, migration_matrix,
            population, base_contact_matrix, mobility_rate)
        k4_S, k4_E, k4_I, k4_R, inc4 = compute_derivatives_seir(
            S_curr + sub_dt * k3_S, E_curr + sub_dt * k3_E,
            I_curr + sub_dt * k3_I, R_curr + sub_dt * k3_R,
            beta, gamma, sigma_incubation, contact_mod, migration_matrix,
            population, base_contact_matrix, mobility_rate)

        S_curr = S_curr + (sub_dt / 6.0) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E_curr = E_curr + (sub_dt / 6.0) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        I_curr = I_curr + (sub_dt / 6.0) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        R_curr = R_curr + (sub_dt / 6.0) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)

        # SEIR模型中，新发报告病例对应于从潜伏期(E)进入感染期(I)的人数(incubation_rate)
        incidence_final += (sub_dt / 6.0) * (inc1 + 2 * inc2 + 2 * inc3 + inc4)

    return (torch.clamp(S_curr, min=0), torch.clamp(E_curr, min=0),
            torch.clamp(I_curr, min=0), torch.clamp(R_curr, min=0),
            torch.clamp(incidence_final, min=0))


def calculate_R0_NGM(beta, gamma, contact_mod, migration_matrix, mobility_rate=0.01):
    """
    【学术重构】：基于严格的多斑块 SEIR 系统的下一代矩阵(NGM)谱半径计算。
    符合顶级传染病动力学期刊的理论金标准 (Diekmann et al.)
    """
    B_dim, N_dim = beta.shape
    device = beta.device
    mu = 1.0 / (70 * 365)
    lambda_dis = 0.00008
    sigma_incubation = 1.0 / 3.0  # 必须与 age_structured_seir_step 中的潜伏期常数保持一致

    transmission_rates = beta * contact_mod
    # F 矩阵：仅 I 仓室对 E 仓室产生新发感染
    F_12 = torch.diag_embed(transmission_rates)

    effective_mig = mobility_rate * migration_matrix
    mask = torch.eye(N_dim, device=device).unsqueeze(0).bool()
    effective_mig = effective_mig.masked_fill(mask, 0.0)
    outflow_rates = torch.sum(effective_mig, dim=-1)

    # 1. 潜伏期 E 仓室的转移矩阵 V_E
    diag_V_E = torch.diag_embed(torch.full((B_dim, N_dim), sigma_incubation + mu, device=device) + outflow_rates)
    off_diag_V_E = effective_mig.transpose(-2, -1)  # 迁入为负
    V_E = diag_V_E - off_diag_V_E + torch.eye(N_dim, device=device).unsqueeze(0) * 1e-5
    V_E_inv = torch.linalg.inv(V_E)

    # 2. 感染期 I 仓室的转移矩阵 V_I
    diag_V_I = torch.diag_embed(gamma + mu + lambda_dis + outflow_rates)
    off_diag_V_I = effective_mig.transpose(-2, -1)
    V_I = diag_V_I - off_diag_V_I + torch.eye(N_dim, device=device).unsqueeze(0) * 1e-5
    V_I_inv = torch.linalg.inv(V_I)

    # 3. 计算 SEIR 系统的 NGM 核心特征块: K = F_12 * V_I^{-1} * sigma * V_E^{-1}
    sigma_matrix = torch.diag_embed(torch.full((B_dim, N_dim), sigma_incubation, device=device))
    K = torch.bmm(torch.bmm(F_12, V_I_inv), torch.bmm(sigma_matrix, V_E_inv))

    eigenvalues = torch.linalg.eigvals(K)
    return torch.max(torch.abs(eigenvalues), dim=-1).values  # 谱半径
