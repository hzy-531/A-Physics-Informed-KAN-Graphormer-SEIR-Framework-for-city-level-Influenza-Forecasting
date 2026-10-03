import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 扩散距离计算
# ==========================================
def compute_diffusion_distance(adj_matrix, t=3):
    """
    计算基于随机游走的扩散距离
    adj_matrix: (N, N) 归一化的迁徙矩阵 (行和为1)
    """
    N = adj_matrix.shape[0]

    if isinstance(adj_matrix, torch.Tensor):
        P = adj_matrix.clone().detach().float()
    else:
        P = torch.tensor(adj_matrix, dtype=torch.float32)

    eig_vals, eig_vecs = torch.linalg.eig(P.T)
    idx = torch.argmin(torch.abs(eig_vals.real - 1.0))
    pi = eig_vecs[:, idx].real
    pi = pi / pi.sum()

    Pt = torch.matrix_power(P, t)

    w = 1.0 / (pi + 1e-8)
    Pt_scaled = Pt * torch.sqrt(w)

    dist_sq = torch.sum((Pt_scaled[:, None, :] - Pt_scaled[None, :, :]) ** 2, dim=2)
    diffusion_dist = torch.sqrt(dist_sq)

    return diffusion_dist


# ==========================================
# 2. 数据集构建
# ==========================================
class EpidemicDataset(Dataset):
    def __init__(self, features, migration_matrices, lookback_window=7, predict_window=3):
        self.features = torch.FloatTensor(features)
        self.migration_matrices = torch.FloatTensor(migration_matrices)
        self.lookback = lookback_window
        self.predict = predict_window

    def __len__(self):
        return len(self.features) - self.lookback - self.predict + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.lookback]
        adj = self.migration_matrices[idx + self.lookback - 1]
        y = self.features[idx + self.lookback: idx + self.lookback + self.predict, :, 0]
        future_adj = self.migration_matrices[idx + self.lookback: idx + self.lookback + self.predict]
        return x, adj, future_adj, y


# ==========================================
# 3. 修复后的 KAN 与 Graphormer 模型组件
# ==========================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # === 修复核心 ===
        # 使用 linspace 替代 arange，确保网格点数量严格可控
        # 网格范围: [-1, 1] 向外扩展 order * h
        h = (1 - (-1)) / grid_size
        num_grid_points = grid_size + 2 * spline_order + 1
        grid = torch.linspace(
            -1 - spline_order * h,
            1 + spline_order * h,
            num_grid_points
        )
        # =============

        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.base_activation = nn.SiLU()
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))

        self.reset_parameters(scale_noise, scale_base, scale_spline)

    def reset_parameters(self, scale_noise, scale_base, scale_spline):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features,
                                self.out_features) - 0.5) * scale_noise / self.grid_size
            self.spline_weight.data.copy_(noise.permute(2, 1, 0) * scale_spline)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:-(k + 1)]) / (grid[k:-1] - grid[:-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[k + 1:] - x) / (grid[k + 1:] - grid[1:-k]) * bases[:, :, 1:]
        return bases

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_norm = torch.tanh(x)
        bases = self.b_splines(x_norm)

        # 此时 bases 的形状应该是 (Batch*Nodes, In, Grid+Order)
        # 如果因为浮点误差导致 bases 多了一维，强制截断以匹配权重
        if bases.size(2) > self.spline_weight.size(2):
            bases = bases[:, :, :self.spline_weight.size(2)]

        spline_output = torch.einsum("big,oig->bo", bases, self.spline_weight)
        output = base_output + spline_output
        return output.view(*original_shape[:-1], self.out_features)


class SpatialEncoder(nn.Module):
    def __init__(self, num_heads, max_dist_bins=100):
        super().__init__()
        self.dist_embedding = nn.Embedding(max_dist_bins, num_heads)

    def forward(self, diff_dist_matrix):
        dist_idx = (diff_dist_matrix * 10).long().clamp(0, 99)
        bias = self.dist_embedding(dist_idx).permute(2, 0, 1).unsqueeze(0)
        return bias


class KANGraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.kan_ffn = nn.Sequential(
            KANLinear(hidden_dim, hidden_dim * 2),
            KANLinear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        kan_out = self.kan_ffn(x)
        x = self.norm2(x + kan_out)
        return x


class KAN_Graphormer_Model(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.spatial_encoder = SpatialEncoder(num_heads=4)
        self.layers = nn.ModuleList([
            KANGraphormerLayer(hidden_dim, num_heads=4) for _ in range(num_layers)
        ])
        self.param_head = nn.Sequential(
            KANLinear(hidden_dim, hidden_dim // 2),
            KANLinear(hidden_dim // 2, 2),
            nn.Softplus()
        )

    def forward(self, x, diffusion_dist):
        x_current = x[:, -1, :, :]
        h = self.input_proj(x_current)
        for layer in self.layers:
            h = layer(h)
        params = self.param_head(h)
        return params[:, :, 0], params[:, :, 1]


# ==========================================
# 4. 动力学演化与数据加载
# ==========================================
def metapopulation_step(S, I, R, beta, gamma, migration_matrix, population):
    new_infections = beta * S * I / (population + 1e-8)
    new_recoveries = gamma * I

    S_next = S - new_infections
    I_next = I + new_infections - new_recoveries
    R_next = R + new_recoveries

    S_final = torch.bmm(migration_matrix, S_next.unsqueeze(-1)).squeeze(-1)
    I_final = torch.bmm(migration_matrix, I_next.unsqueeze(-1)).squeeze(-1)
    R_final = torch.bmm(migration_matrix, R_next.unsqueeze(-1)).squeeze(-1)

    return S_final, I_final, R_final


def load_real_data(flu_path, feature_path, migration_path):
    print("正在加载数据...")
    df_flu = pd.read_excel(flu_path)
    df_feat = pd.read_excel(feature_path)

    cities = sorted(df_flu['城市名称 (City_Name)'].unique())
    dates = sorted(df_flu['日期 (Date)'].unique())
    num_nodes = len(cities)

    city_to_idx = {city: i for i, city in enumerate(cities)}

    city_static_feats = []
    real_populations = []

    for city in cities:
        row = df_feat[df_feat['城市名称'] == city]
        if len(row) == 0:
            gdp, pop, region = 0, 0, 0
        else:
            gdp = row['2019GDP(亿元)'].values[0]
            pop = row['2019市常住人口(万人)'].values[0] * 10000
            region = row['区域标识 (Region)(南方=1，北方=0)'].values[0]

        city_static_feats.append([gdp, pop, region])
        real_populations.append(pop)

    static_feats = np.array(city_static_feats)
    static_feats[:, :2] = (static_feats[:, :2] - static_feats[:, :2].mean(axis=0)) / (
                static_feats[:, :2].std(axis=0) + 1e-8)

    features_list = []
    for date in dates:
        day_feat = np.zeros((num_nodes, 4))
        day_feat[:, 1:] = static_feats

        current_df = df_flu[df_flu['日期 (Date)'] == date]
        for _, row in current_df.iterrows():
            city = row['城市名称 (City_Name)']
            if city in city_to_idx:
                case = row['城市_日估算病例 (City_Daily_Case)']
                day_feat[city_to_idx[city], 0] = case
        features_list.append(day_feat)

    features_tensor = np.array(features_list)
    max_cases = features_tensor[:, :, 0].max()
    features_tensor[:, :, 0] = features_tensor[:, :, 0] / (max_cases + 1e-8)

    print(f"正在加载迁徙矩阵: {migration_path}")
    xls = pd.ExcelFile(migration_path)
    adj_list = []
    for date in dates:
        date_str = pd.to_datetime(date).strftime("D%Y%m%d")
        if date_str in xls.sheet_names:
            df_mat = pd.read_excel(xls, sheet_name=date_str, index_col=0)
            df_mat = df_mat.reindex(index=cities, columns=cities, fill_value=0)
            adj_list.append(df_mat.values)
        else:
            adj_list.append(np.eye(num_nodes))

    return features_tensor, np.array(adj_list), max_cases, np.array(real_populations)


# ==========================================
# 5. 主训练循环 (增强版：含学习率衰减、最优保存、可视化)
# ==========================================
def train_model():
    # 配置路径
    path_flu = r"E:\Claude code\KAN+\data\原始数据\01_Influenza_Target_Filled.xlsx"
    path_feat = r"E:\Claude code\KAN+\data\原始数据\02_City_Features.xlsx"
    path_mig = r"E:\Claude code\KAN+\data\原始数据\03_Migration_Matrices_Outflow_Normalized.xlsx"

    if not os.path.exists(path_flu):
        print(f"错误: 找不到文件 {path_flu}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    real_features, real_adj, max_case_val, real_pops = load_real_data(path_flu, path_feat, path_mig)

    # 2. 计算扩散距离
    mean_adj = torch.tensor(real_adj.mean(axis=0), dtype=torch.float32)
    diffusion_dist = compute_diffusion_distance(mean_adj).to(device)

    # 3. 数据集
    dataset = EpidemicDataset(real_features, real_adj)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 模型
    num_nodes = real_features.shape[1]
    features_dim = real_features.shape[2]
    model = KAN_Graphormer_Model(num_nodes, features_dim, hidden_dim=64).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # === 改进1: 学习率调度器 (每10轮衰减为原来的0.8) ===
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    population = torch.tensor(real_pops, dtype=torch.float32).to(device).unsqueeze(0)

    # 用于记录Loss画图
    loss_history = []
    best_loss = float('inf')
    best_model_path = r"E:\Claude code\KAN+\result\best_kan_model.pth"

    print("开始训练...")
    for epoch in range(50):
        model.train()
        total_loss = 0

        for x, adj_past, adj_future, y_true in dataloader:
            x, y_true = x.to(device), y_true.to(device)
            adj_future = adj_future.to(device)

            optimizer.zero_grad()

            # 模型预测 SIR 参数
            beta, gamma = model(x, diffusion_dist)

            # 物理演化
            I_curr = x[:, -1, :, 0] * max_case_val
            S_curr = torch.clamp(population - I_curr, min=0)
            R_curr = torch.zeros_like(I_curr)

            predictions = []
            for t in range(y_true.shape[1]):
                mig_mat = adj_future[:, t]
                S_curr, I_curr, R_curr = metapopulation_step(
                    S_curr, I_curr, R_curr, beta, gamma, mig_mat, population
                )
                predictions.append(I_curr)

            pred_tensor = torch.stack(predictions, dim=1)
            pred_norm = pred_tensor / (max_case_val + 1e-8)

            loss = F.mse_loss(pred_norm, y_true)

            loss.backward()

            # === 改进2: 梯度裁剪 (防止 Loss 突然暴涨) ===
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        scheduler.step()  # 更新学习率

        # === 改进3: 保存最优模型 ===
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch + 1}: Loss下降至 {avg_loss:.6f} [已保存最优模型]")
        else:
            print(f"Epoch {epoch + 1}: Loss: {avg_loss:.6f}")

    print(f"训练完成！最优 Loss: {best_loss:.6f}")

    # === 改进4: 简单的可视化验证 (画出 Loss 曲线) ===
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('KAN-Graphormer Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'E:\Claude code\KAN+\result\training_loss_curve.png')
    print("Loss 曲线已保存为 training_loss_curve.png")


if __name__ == "__main__":
    train_model()