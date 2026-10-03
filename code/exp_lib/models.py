"""
模型定义模块 — 从 exp_chronological.py 提取
包含全部 7 个消融实验模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers import (RobustKANLinear, StandardMLP, GatedTCN,
                     GraphormerLayer, ImprovedGATLayer)
from .physics import age_structured_seir_step


# ==========================================
# 1. EnhancedFull_Graphormer_V3 — 完整 KAN-Graphormer
# ==========================================

class EnhancedFull_Graphormer_V3(nn.Module):
    """统一的主干网络：通过传参控制消融属性"""

    def __init__(self, num_nodes=24, input_dim=16, hidden_dim=256, num_heads=8,
                 num_layers=2, lookback=7, use_tcn=True, enable_physics=True,
                 use_kan=True, use_kan_ffn=None, predict_window=3,
                 tcn_dropout=0.15, graphormer_dropout=0.2,
                 kan_grid_size=3, kan_scale_spline=0.001):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.predict_window = predict_window
        self.use_tcn = use_tcn
        self.enable_physics = enable_physics

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.in_degree_embed = nn.Embedding(num_nodes, hidden_dim)
        self.out_degree_embed = nn.Embedding(num_nodes, hidden_dim)

        if use_tcn:
            self.tcn1 = GatedTCN(hidden_dim, kernel_size=3, dilation=1, dropout=tcn_dropout)
            self.tcn2 = GatedTCN(hidden_dim, kernel_size=3, dilation=2, dropout=tcn_dropout)
        else:
            self.tcn1 = nn.Identity()
            self.tcn2 = nn.Identity()

        # use_kan_ffn 控制 Graphormer FFN 是否用 KAN (默认跟随 use_kan)
        if use_kan_ffn is None:
            use_kan_ffn = use_kan

        self.graphormer_layers = nn.ModuleList()
        self.dense_gates = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.5])) for _ in range(num_layers)])
        self.lockdown_alpha = nn.Parameter(torch.tensor([0.5]))
        self.region_beta_boost = nn.Parameter(torch.tensor([0.1]))
        for _ in range(num_layers):
            self.graphormer_layers.append(
                GraphormerLayer(hidden_dim, num_heads, dropout=graphormer_dropout, use_kan=use_kan_ffn))

        if use_kan:
            self.case_head = nn.Sequential(
                RobustKANLinear(hidden_dim, hidden_dim // 2, grid_size=kan_grid_size,
                                scale_base=1.0, scale_spline=kan_scale_spline),
                RobustKANLinear(hidden_dim // 2, self.predict_window * 3,
                                grid_size=kan_grid_size, scale_base=1.0, scale_spline=kan_scale_spline)
            )
        else:
            self.case_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                nn.Linear(hidden_dim // 2, self.predict_window * 3)
            )
            nn.init.constant_(self.case_head[-1].bias, 0.0)
            nn.init.xavier_normal_(self.case_head[-1].weight, gain=0.1)

        if self.enable_physics:
            if use_kan:
                self.beta_head = nn.Sequential(
                    RobustKANLinear(hidden_dim, hidden_dim // 2, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline),
                    RobustKANLinear(hidden_dim // 2, self.predict_window,
                                    grid_size=kan_grid_size, scale_base=1.0, scale_spline=kan_scale_spline)
                )
                self.gamma_head = nn.Sequential(
                    RobustKANLinear(hidden_dim, hidden_dim // 2, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline),
                    RobustKANLinear(hidden_dim // 2, 1, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline)
                )
                self.contact_head = nn.Sequential(
                    RobustKANLinear(hidden_dim, hidden_dim // 2, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline),
                    RobustKANLinear(hidden_dim // 2, self.predict_window,
                                    grid_size=kan_grid_size, scale_base=1.0, scale_spline=kan_scale_spline)
                )
                self.i0_head = nn.Sequential(
                    RobustKANLinear(hidden_dim, hidden_dim // 2, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline),
                    RobustKANLinear(hidden_dim // 2, 1, grid_size=kan_grid_size,
                                    scale_base=1.0, scale_spline=kan_scale_spline)
                )
            else:
                self.beta_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                    nn.Linear(hidden_dim // 2, self.predict_window))
                self.gamma_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1))
                self.contact_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                    nn.Linear(hidden_dim // 2, self.predict_window))
                self.i0_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, adj_seq, dist_matrix, time_step=None):
        B, L, N, feat_dim = x.shape
        x_last = x[:, -1, :, :]
        h = self.input_proj(x_last)

        adj_cur = adj_seq[:, 0, :, :] if adj_seq is not None and adj_seq.dim() == 4 else adj_seq
        if adj_cur is not None:
            adj_binary = (adj_cur > 1e-4).float()
            in_deg = adj_binary.sum(dim=2).long().clamp(0, self.num_nodes - 1)
            out_deg = adj_binary.sum(dim=1).long().clamp(0, self.num_nodes - 1)
            h = h + self.in_degree_embed(in_deg) + self.out_degree_embed(out_deg)

        if self.use_tcn:
            x_seq = x.permute(0, 2, 1, 3).reshape(B * N, L, feat_dim)
            x_seq_proj = self.input_proj(x_seq).transpose(1, 2)
            tcn_out = self.tcn2(self.tcn1(x_seq_proj)) + x_seq_proj
            h = h + tcn_out.transpose(1, 2)[:, -1, :].reshape(B, N, self.hidden_dim)

        layer_outputs = [h]
        attn_weights_list = []
        gate = torch.sigmoid(self.dense_gates[0])
        for i, layer in enumerate(self.graphormer_layers):
            h_in = h if i == 0 else gate * h + (1.0 - gate) * torch.stack(layer_outputs).mean(dim=0)
            h, attn_w = layer(h_in, adj_cur, dist_matrix)
            gate = torch.sigmoid(self.dense_gates[i])
            layer_outputs.append(h)
            attn_weights_list.append(attn_w)

        h = gate * h + (1.0 - gate) * torch.stack(layer_outputs).mean(dim=0)

        cases_pred_raw = self.case_head(h)
        cases_pred_raw = cases_pred_raw.view(B, N, self.predict_window, 3).transpose(1, 2)
        last_day = x[:, -1, :, 0:3].unsqueeze(1)
        cases_pred_nn = torch.clamp(last_day + cases_pred_raw, 0.0, 1.2)

        if self.enable_physics:
            h_phys = h
            beta_raw = self.beta_head(h_phys)
            if beta_raw.dim() == 2:
                beta_raw = beta_raw.unsqueeze(-1)
            region_flag = x[:, -1, :, 10].unsqueeze(-1)
            beta = torch.sigmoid(beta_raw + F.softplus(self.region_beta_boost) * region_flag) * 0.9 + 0.1
            base_contact_mod = torch.sigmoid(self.contact_head(h_phys)) * 1.5 + 0.1
            lockdown_factor = x[:, -1, :, 9].unsqueeze(-1)
            contact_mod = base_contact_mod * torch.exp(-F.softplus(self.lockdown_alpha) * lockdown_factor)
            gamma = torch.sigmoid(self.gamma_head(h_phys).squeeze(-1)) * (0.333 - 0.142) + 0.142
            i0_mult = torch.sigmoid(self.i0_head(h_phys).squeeze(-1)) * 4.0 + 1.0
        else:
            beta = gamma = base_contact_mod = contact_mod = i0_mult = None

        return (cases_pred_nn, beta, gamma, base_contact_mod, contact_mod, i0_mult,
                attn_weights_list[-1] if attn_weights_list else None)


# ==========================================
# 2. EnhancedMLP_Baseline_V3 — 纯 MLP 基线
# ==========================================

class EnhancedMLP_Baseline_V3(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=32, output_dim=3, lookback=7, predict_window=3):
        super().__init__()
        self.lookback = lookback
        self.predict_window = predict_window
        self.flatten_dim = lookback * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, output_dim * predict_window),
        )
        nn.init.constant_(self.net[-1].bias, 0.0)
        nn.init.xavier_normal_(self.net[-1].weight, gain=0.1)

    def forward(self, x, adj=None, dist_matrix=None, time_step=None):
        B, L, N, feat_dim = x.shape
        x_city = x.transpose(1, 2).reshape(B * N, L * feat_dim)
        out = self.net(x_city)
        out = out.view(B, N, self.predict_window, 3).transpose(1, 2)
        last_day = x[:, -1, :, 0:3].unsqueeze(1)
        cases_pred = torch.clamp(last_day + out, 0.0, 1.2)
        return cases_pred, None, None, None, None, None, None


# ==========================================
# 3. EnhancedLSTM_Baseline_V3 — LSTM 基线
# ==========================================

class EnhancedLSTM_Baseline_V3(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=16, num_layers=1, output_dim=3,
                 lookback=7, predict_window=3):
        super().__init__()
        self.lookback = lookback
        self.predict_window = predict_window
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim * predict_window)
        )
        nn.init.constant_(self.fc[-1].bias, 0.0)
        nn.init.xavier_normal_(self.fc[-1].weight, gain=0.1)

    def forward(self, x, adj=None, dist_matrix=None, time_step=None):
        B, L, N, feat_dim = x.shape
        x_city = x.transpose(1, 2).reshape(B * N, L, feat_dim)
        lstm_out, _ = self.lstm(x_city)
        out = self.fc(lstm_out[:, -1, :])
        out = out.view(B, N, self.predict_window, 3).transpose(1, 2)
        last_day = x[:, -1, :, 0:3].unsqueeze(1)
        cases_pred = torch.clamp(last_day + out, 0.0, 1.2)
        return cases_pred, None, None, None, None, None, None


# ==========================================
# 4. EnhancedGAT_Baseline_V3 — GAT 基线
# ==========================================

class EnhancedGAT_Baseline_V3(nn.Module):
    def __init__(self, in_features=112, hidden_dim=64, heads=2, output_dim=3,
                 num_layers=2, lookback=7, predict_window=3):
        super().__init__()
        self.in_features = in_features
        self.predict_window = predict_window
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(0.2))
        self.gat_layers = nn.ModuleList([
            ImprovedGATLayer(hidden_dim,
                             hidden_dim // heads if i < num_layers - 1 else hidden_dim,
                             heads, 0.2, i < num_layers - 1)
            for i in range(num_layers)])
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.case_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim * predict_window)
        )
        nn.init.constant_(self.case_head[-1].bias, 0.0)
        nn.init.xavier_normal_(self.case_head[-1].weight, gain=0.1)

    def forward(self, x, adj, dist_matrix=None, time_step=None):
        B, L, N, feat_dim = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        h = self.input_proj(x_flat)
        adj_cur = adj[:, 0, :, :] if adj.dim() == 4 else adj
        final_attn = None
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norm_layers)):
            h_new, attn = gat(h, adj_cur)
            h = norm(h + h_new)
            if i == len(self.gat_layers) - 1:
                final_attn = attn
        out = self.case_head(h).view(B, N, self.predict_window, 3).transpose(1, 2)
        last_day = x[:, -1, :, 0:3].unsqueeze(1)
        cases_pred = torch.clamp(last_day + out, 0.0, 1.2)
        return cases_pred, None, None, None, None, None, final_attn


# ==========================================
# 5. EnhancedKAN_Only_V3 — 纯 KAN 消融模型
# ==========================================

class EnhancedKAN_Only_V3(nn.Module):
    """真正的纯 KAN 消融模型：无图结构，无注意力，完全依赖时序非线性映射"""

    def __init__(self, input_dim=16, hidden_dim=192, use_tcn=False, predict_window=3):
        super().__init__()
        self.lookback = 7
        self.predict_window = predict_window
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim * self.lookback, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2), nn.Dropout(0.1)
        )
        self.kan_layer = RobustKANLinear(hidden_dim // 2, hidden_dim // 2,
                                         grid_size=6, spline_order=3)
        self.lockdown_alpha = nn.Parameter(torch.tensor([0.5]))
        self.region_beta_boost = nn.Parameter(torch.tensor([0.1]))

        self.beta_head = nn.Sequential(
            RobustKANLinear(hidden_dim // 2, hidden_dim // 4, grid_size=5),
            nn.LayerNorm(hidden_dim // 4),
            RobustKANLinear(hidden_dim // 4, self.predict_window, grid_size=5,
                            scale_base=0.1, scale_spline=0.01)
        )
        self.gamma_head = nn.Sequential(
            RobustKANLinear(hidden_dim // 2, hidden_dim // 4, grid_size=5),
            nn.LayerNorm(hidden_dim // 4),
            RobustKANLinear(hidden_dim // 4, 1, grid_size=5,
                            scale_base=0.1, scale_spline=0.01)
        )
        self.contact_head = nn.Sequential(
            RobustKANLinear(hidden_dim // 2, hidden_dim // 4, grid_size=5),
            nn.LayerNorm(hidden_dim // 4),
            RobustKANLinear(hidden_dim // 4, self.predict_window, grid_size=5,
                            scale_base=0.1, scale_spline=0.01)
        )
        self.i0_head = nn.Sequential(
            RobustKANLinear(hidden_dim // 2, hidden_dim // 4, grid_size=5),
            nn.LayerNorm(hidden_dim // 4),
            RobustKANLinear(hidden_dim // 4, 1, grid_size=5,
                            scale_base=0.1, scale_spline=0.01)
        )
        self.case_head = nn.Sequential(
            RobustKANLinear(hidden_dim // 2, hidden_dim // 4, grid_size=5),
            nn.LayerNorm(hidden_dim // 4),
            RobustKANLinear(hidden_dim // 4, self.predict_window * 3, grid_size=5,
                            scale_base=0.1, scale_spline=0.01)
        )

    def forward(self, x, adj=None, dist_matrix=None, time_step=None):
        B, L, N, feat_dim = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L * feat_dim)
        h = self.kan_layer(self.temporal_encoder(x_flat).reshape(B, N, -1))
        h_reshaped = h.reshape(B * N, -1)

        cases_pred_raw = self.case_head(h_reshaped)
        cases_pred_raw = cases_pred_raw.view(B, N, self.predict_window, 3).transpose(1, 2)
        last_day = x[:, -1, :, 0:3].unsqueeze(1)
        cases_pred_nn = torch.clamp(last_day + cases_pred_raw, 0.0, 1.2)

        h_phys = h_reshaped
        beta_raw = self.beta_head(h_phys).reshape(B, N, self.predict_window)
        region_flag = x[:, -1, :, 10].unsqueeze(-1)
        beta = torch.sigmoid(beta_raw + F.softplus(self.region_beta_boost) * region_flag) * 0.9 + 0.1
        base_contact_mod = torch.sigmoid(
            self.contact_head(h_phys).reshape(B, N, self.predict_window)) * 1.5 + 0.1
        lockdown_factor = x[:, -1, :, 9].unsqueeze(-1)
        contact_mod = base_contact_mod * torch.exp(
            -F.softplus(self.lockdown_alpha) * lockdown_factor)
        gamma = torch.sigmoid(self.gamma_head(h_phys).reshape(B, N)) * (0.333 - 0.142) + 0.142
        i0_mult = torch.sigmoid(self.i0_head(h_phys).reshape(B, N)) * 4.0 + 1.0

        return cases_pred_nn, beta, gamma, base_contact_mod, contact_mod, i0_mult, None


# ==========================================
# 6. M_Graphormer_Baseline — MLP-Graphormer (无 KAN)
# ==========================================

class M_Graphormer_Baseline(EnhancedFull_Graphormer_V3):
    """MLP-Graphormer 基线: 图结构 + MLP FFN + 物理, 无 KAN"""

    def __init__(self, num_nodes=24, input_dim=16, hidden_dim=256, num_heads=8,
                 num_layers=2, lookback=7, predict_window=3, **kwargs):
        super().__init__(num_nodes=num_nodes, input_dim=input_dim, hidden_dim=hidden_dim,
                         num_heads=num_heads, num_layers=num_layers, lookback=lookback,
                         use_tcn=True, enable_physics=True, use_kan=False,
                         predict_window=predict_window)


# ==========================================
# 7. NoPhysics_KAN_Graphormer — KAN-Graphormer 无物理
# ==========================================

class NoPhysics_KAN_Graphormer(EnhancedFull_Graphormer_V3):
    """KAN-Graphormer 无物理: 图结构 + KAN FFN, 关闭物理 ODE"""

    def __init__(self, num_nodes=24, input_dim=16, hidden_dim=256, num_heads=8,
                 num_layers=2, lookback=7, predict_window=3, **kwargs):
        super().__init__(num_nodes=num_nodes, input_dim=input_dim, hidden_dim=hidden_dim,
                         num_heads=num_heads, num_layers=num_layers, lookback=lookback,
                         use_tcn=True, enable_physics=False, use_kan=True,
                         predict_window=predict_window)


# ==========================================
# 模型注册表
# ==========================================

MODEL_DEFINITIONS = {
    'MLP_Baseline': {
        'class': EnhancedMLP_Baseline_V3,
        'requires_adj': False, 'requires_dist': False, 'requires_full_seq': False,
        'has_physics': False, 'has_kan': False,
        'display_name': 'MLP Baseline'
    },
    'LSTM_Baseline': {
        'class': EnhancedLSTM_Baseline_V3,
        'requires_adj': False, 'requires_dist': False, 'requires_full_seq': False,
        'has_physics': False, 'has_kan': False,
        'display_name': 'LSTM Baseline'
    },
    'GAT_Baseline': {
        'class': EnhancedGAT_Baseline_V3,
        'requires_adj': True, 'requires_dist': False, 'requires_full_seq': False,
        'has_physics': False, 'has_kan': False,
        'display_name': 'GAT Baseline'
    },
    'M_Graphormer_Baseline': {
        'class': M_Graphormer_Baseline,
        'requires_adj': True, 'requires_dist': True, 'requires_full_seq': True,
        'has_physics': True, 'has_kan': False,
        'display_name': 'MLP-Graphormer Baseline'
    },
    'KAN_Only': {
        'class': EnhancedKAN_Only_V3,
        'requires_adj': False, 'requires_dist': False, 'requires_full_seq': False,
        'has_physics': True, 'has_kan': True,
        'display_name': 'KAN Only'
    },
    'NoPhysics_KAN_Graphormer': {
        'class': NoPhysics_KAN_Graphormer,
        'requires_adj': True, 'requires_dist': True, 'requires_full_seq': True,
        'has_physics': False, 'has_kan': True,
        'display_name': 'No-Physics KAN-Graphormer'
    },
    'Full_KAN_Graphormer': {
        'class': EnhancedFull_Graphormer_V3,
        'requires_adj': True, 'requires_dist': True, 'requires_full_seq': True,
        'has_physics': True, 'has_kan': True,
        'display_name': 'Full KAN-Graphormer'
    },
}
