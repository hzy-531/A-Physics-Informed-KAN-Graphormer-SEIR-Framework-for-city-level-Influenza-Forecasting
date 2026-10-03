"""
神经网络基础层模块 — 从 exp_chronological.py 提取
包含 KAN、TCN、GAT、Graphormer、空间编码等可复用组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ==========================================
# 工具函数
# ==========================================

def compute_diffusion_distance(adj_matrix, t=3):
    """扩散距离计算 — 用于空间编码"""
    if isinstance(adj_matrix, torch.Tensor):
        P = adj_matrix.clone().detach().float()
    else:
        P = torch.tensor(adj_matrix, dtype=torch.float32)

    row_sum = P.sum(dim=1, keepdim=True)
    row_sum_zero = (row_sum.squeeze() == 0)
    P = P / (row_sum + 1e-8)
    if row_sum_zero.any():
        P[row_sum_zero] = 0.0
        P[row_sum_zero, torch.arange(P.shape[1])[row_sum_zero]] = 1.0

    try:
        eig_vals, eig_vecs = torch.linalg.eig(P.T)
        idx = torch.argmin(torch.abs(eig_vals.real - 1.0))
        pi = eig_vecs[:, idx].real
        pi = pi / (pi.sum() + 1e-8)
    except:
        pi = torch.ones(P.size(-1), device=P.device) / P.size(-1)

    Pt = torch.matrix_power(P, t)
    Pt_scaled = Pt / torch.sqrt(pi + 1e-8)
    Pt_sq = torch.sum(Pt_scaled ** 2, dim=1, keepdim=True)
    dist_sq = Pt_sq + Pt_sq.t() - 2 * torch.mm(Pt_scaled, Pt_scaled.t())

    return torch.sqrt(torch.clamp(dist_sq, min=1e-8))


# ==========================================
# RobustKANLinear — KAN 2.0 原理的稳健实现
# ==========================================

class RobustKANLinear(nn.Module):
    """基于 KAN 2.0 原理的稳健实现 (防方差爆炸版)"""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order

        # 网格覆盖 LayerNorm 后的高斯分布 [-3, 3] 区间 (涵盖 99.7% 的数据)
        h = 6.0 / grid_size
        grid = torch.linspace(-3.0 - spline_order * h, 3.0 + (spline_order + 1) * h,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            std = scale_noise / math.sqrt(in_features + grid_size)
            noise = torch.randn(out_features, in_features,
                                grid_size + spline_order) * std
            self.spline_weight.data.copy_(noise * scale_spline)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid.to(x.device)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            d_left = grid[k:-1] - grid[:-(k + 1)]
            d_right = grid[k + 1:] - grid[1:-k]
            d_left = torch.where(d_left == 0, torch.ones_like(d_left), d_left)
            d_right = torch.where(d_right == 0, torch.ones_like(d_right), d_right)
            b_left = bases[..., :-1]
            b_right = bases[..., 1:]
            bases = (x - grid[:-(k + 1)]) / d_left * b_left + \
                    (grid[k + 1:] - x) / d_right * b_right
        return bases

    def forward(self, x):
        x_norm_clamped = 3.0 * torch.tanh(x / 3.0)
        base_out = F.linear(self.base_activation(x), self.base_weight)
        bases = self.b_splines(x_norm_clamped)
        if bases.size(-1) > self.spline_weight.size(2):
            bases = bases[..., :self.spline_weight.size(2)]
        spline_out = torch.einsum("...ig,oig->...o", bases, self.spline_weight)
        return base_out + spline_out


# ==========================================
# StandardMLP — 对标 KAN 参数量的普通 MLP
# ==========================================

class StandardMLP(nn.Module):
    """用于消融实验的对标MLP，参数量与 RobustKANLinear 对齐"""

    def __init__(self, in_features, out_features):
        super().__init__()
        hidden = in_features * 4
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# GatedTCN — 门控时序卷积层
# ==========================================

class GatedTCN(nn.Module):
    """门控时序卷积层 - 输入输出通道数相同，保持维度"""

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            bias=True
        )
        self.gate_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            bias=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [batch, channels, seq_len]
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate_conv(x))
        output = conv_out * gate_out
        if output.shape[-1] > x.shape[-1]:
            output = output[:, :, :x.shape[-1]]
        output = output.transpose(1, 2)  # [batch, seq_len, channels]
        output = self.norm(output)
        output = output.transpose(1, 2)  # [batch, channels, seq_len]
        output = self.dropout(output)
        return output


# ==========================================
# RBFSpatialEncoding — 高斯径向基空间编码
# ==========================================

class RBFSpatialEncoding(nn.Module):
    """【学术重构】：全局缩放的高斯径向基空间编码，防止小样本下的图特征过拟合"""

    def __init__(self, num_heads, num_centers=16):
        super().__init__()
        self.num_heads = num_heads
        self.centers = nn.Parameter(torch.linspace(0, 1.0, num_centers))
        self.widths = nn.Parameter(torch.ones(num_centers) * 0.1)
        self.proj = nn.Sequential(
            nn.Linear(num_centers, num_heads * 2),
            nn.SiLU(),
            nn.Linear(num_heads * 2, num_heads)
        )

    def forward(self, dist_matrix):
        max_dist = dist_matrix.max() + 1e-5
        norm_dist = dist_matrix / max_dist
        d = norm_dist.unsqueeze(-1)
        rbf_expansion = torch.exp(-((d - self.centers) ** 2) / (self.widths ** 2 + 1e-5))
        bias = self.proj(rbf_expansion)
        return bias.permute(0, 3, 1, 2)


# ==========================================
# GraphormerLayer — 图 Transformer 层
# ==========================================

class GraphormerLayer(nn.Module):
    """【学术重构】：移除双支路冗余，强制实行单一非线性变换约束"""

    def __init__(self, hidden_dim, num_heads, dropout=0.15, use_kan=True):
        super().__init__()
        self.dropout_rate = dropout
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=self.dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.spatial_encoder = RBFSpatialEncoding(num_heads=num_heads)
        self.edge_proj = nn.Linear(1, num_heads)
        self.use_kan = use_kan

        if use_kan:
            self.ffn = nn.Sequential(
                RobustKANLinear(hidden_dim, hidden_dim * 2, grid_size=5, scale_spline=0.05),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(self.dropout_rate),
                RobustKANLinear(hidden_dim * 2, hidden_dim, grid_size=5, scale_spline=0.05)
            )
        else:
            self.ffn = nn.Sequential(
                StandardMLP(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(self.dropout_rate),
                StandardMLP(hidden_dim * 2, hidden_dim)
            )

        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj, dist_matrix):
        B, N, H = x.shape
        spatial_bias = self.spatial_encoder(dist_matrix)
        edge_bias = self.edge_proj(adj.unsqueeze(-1)).permute(0, 3, 1, 2)

        attn_mask = spatial_bias + edge_bias
        attn_mask = attn_mask.reshape(B * self.attention.num_heads, N, N)

        topology_mask = (adj > 0) | (torch.eye(N, device=adj.device).unsqueeze(0).bool())
        topology_mask = topology_mask.unsqueeze(1).expand(-1, self.attention.num_heads, -1, -1)
        topology_mask = topology_mask.reshape(B * self.attention.num_heads, N, N)

        attn_mask = attn_mask.masked_fill(~topology_mask, float('-inf'))

        residual = x
        x_attn, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(residual + x_attn)

        residual = x
        x_ffn = self.ffn(x)
        x_ffn = F.dropout(x_ffn, p=self.dropout_rate, training=self.training)
        x = self.norm2(residual + x_ffn)
        return x, attn_weights


# ==========================================
# ImprovedGATLayer — 改进的 GAT 层
# ==========================================

class ImprovedGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.Tensor(heads, out_features))
        self.attn_dst = nn.Parameter(torch.Tensor(heads, out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

        if concat:
            self.norm = nn.LayerNorm(heads * out_features)
        else:
            self.norm = nn.LayerNorm(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, adj):
        batch_size, num_nodes, _ = x.shape
        h = self.linear(x)
        h = h.view(batch_size, num_nodes, self.heads, self.out_features)
        h_heads = h.permute(0, 2, 1, 3)

        attn_src = torch.matmul(h_heads, self.attn_src.unsqueeze(-1)).squeeze(-1)
        attn_dst = torch.matmul(h_heads, self.attn_dst.unsqueeze(-1)).squeeze(-1)
        attention_scores = attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2)
        attention_scores = self.leaky_relu(attention_scores)

        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
            adj_mask = adj.unsqueeze(1).expand(-1, self.heads, -1, -1)
            attention_scores = attention_scores.masked_fill(adj_mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        h_weighted = torch.einsum('bhij,bhjc->bhic', attention_weights, h_heads)

        if self.concat:
            output = h_weighted.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        else:
            output = h_weighted.mean(dim=1)

        output = self.norm(output)
        avg_attention = attention_weights.mean(dim=1)
        return output, avg_attention
