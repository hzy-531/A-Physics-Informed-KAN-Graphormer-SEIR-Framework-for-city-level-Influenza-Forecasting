# exp_chronological.py 完整架构解析

> 文件: `experiments/exp_chronological.py` (5058行)
> 作用: KAN-Graphormer 24城市流感传播预测的主程序
> 拆分版本: `exp_lib/` (9模块) + `experiments/run_*.py`

---

## 目录

1. [全局概览](#1-全局概览)
2. [阶段0: 数据加载](#2-阶段0-数据加载)
3. [阶段1: 数据划分与归一化](#3-阶段1-数据划分与归一化)
4. [阶段2A: 模型初始化](#4-阶段2a-模型初始化)
5. [阶段2B: 核心模型前向传播](#5-阶段2b-核心模型前向传播)
6. [阶段3: 训练循环](#6-阶段3-训练循环)
7. [阶段4: 评估与可视化](#7-阶段4-评估与可视化)
8. [附录: 关键公式与常量](#8-附录-关键公式与常量)

---

## 1. 全局概览

整个程序可以分为五大阶段，一条数据从头流到尾：

```
[阶段0] 数据加载      4个Excel → NumPy数组
[阶段1] 数据预处理    滑动窗口 + 归一化 + DataLoader
[阶段2] 模型定义      7个消融模型 + 前向传播
[阶段3] 训练循环      前向→损失→反向→验证→早停
[阶段4] 评估与可视化   R²/RMSE/MAE + 参数反演 + 20+图表
```

### 核心配置常量

| 常量 | 值 | 含义 |
|------|-----|------|
| lookback | 7 | 回看7天历史 |
| predict | 3 | 预测未来3天 |
| train/val/test | 60%/20%/20% | 时序划分比例 |
| hidden_dim | 16 | 隐层维度（微数据集压缩） |
| 24城市 | 北京~石家庄 | 含4一线+8二线+12其他 |
| 特征维度 | 11→17 | 基础11维 + 时序增强6维 |
| SEED | 5780 | 全局随机种子 |

---

## 2. 阶段0: 数据加载

### 函数: `load_raw_data_only()` (L4597-4705)

**输入**: 4个Excel文件

| 文件 | 内容 | 形状 |
|------|------|------|
| `01_Influenza_Target_Filled.xlsx` | 流感日估算病例 | 3264行 × 11列 |
| `02_City_Features.xlsx` | 城市静态特征 | 24行 × 11列 |
| `03_Migration_Matrices_Outflow_Normalized.xlsx` | 城间迁徙矩阵 | 136 sheet × 24×24 |
| `Final_Thesis_Dataset_2020.xlsx` | 环境/气象/搜索 | 3264行 × 15列 |

### 处理流程

```
步骤1: 读取流感病例
  df_flu = pd.read_excel(流感文件)
  '病例_原始' = clip(城市日估算病例, 0, ∞)

步骤2: 读取并清洗环境数据
  df_env = pd.read_excel(环境文件)
  preprocess_intra_city_data(df_env)
    → 公交/地铁缺失值: 城市内线性插值 + 中位数回填

步骤3: 计算年龄比例
  get_smooth_age_ratios(dates)
  → 三次样条插值: 1月→2月→3月 平滑过渡
  → 每日: [儿童%, 成人%, 老年%], 和=1.0

步骤4: 逐日逐城构建基础特征 (11维)
  对每天(136天) × 每城(24城):
    [0] 儿童病例 = 总病例 × 儿童比例
    [1] 成人病例 = 总病例 × 成人比例
    [2] 老年病例 = 总病例 × 老年比例
    [3] 公交百度搜索指数
    [4] 地铁百度搜索指数
    [5] 最终气温(℃)
    [6] 人口密度 (城市静态)
    [7] 城市等级 1=一线, 2=二线, 3=其他
    [8] 供暖标识 0/1
    [9] 封城强度 = compute_lockdown_factor(城市, 日期)
        → 封城前=0, 封城中=强度×渐进因子, 封城后=强度×衰减因子
    [10] 区域标识 南方=1, 北方=0

步骤5: 时序增强 (11维→17维)
  TemporalFeatureEnhancer.add_temporal_features():
    [11] 病例增长率 (当天/前一天-1, 截断[-0.8, 2.0])
    [12] 7日移动平均
    [13] 当前/移动平均比值 (截断[0, 3.0])
    [14] 7日移动标准差
    [15] 周度因子 (0.8~0.9, 周末略低)
    [16] 疫情阶段 (0=上升期, 0.5=平稳期, 1=下降期)

步骤6: 加载迁徙矩阵
  136天 × 24×24, 行和=1.0, 对角线=0
  缺失日期用前一天填充

步骤7: 构建人口张量
  pops (24, 3): 儿童15%, 成人70%, 老年15%
```

### 输出

| 变量 | 形状 | 含义 |
|------|------|------|
| `features_enhanced` | (136, 24, 17) | 17维增强特征 |
| `adjs` | (136, 24, 24) | 日迁徙矩阵 |
| `pops` | (24, 3) | 年龄分层人口 |
| `cities` | (24,) | 城市名称列表 |
| `dates` | (136,) | 日期列表 |

---

## 3. 阶段1: 数据划分与归一化

### 函数: `main()` 前半部分 (L4706-4796)

### 3.1 滑动窗口划分

```
总天数: 136
滑动窗口: lookback=7, predict=3
总样本数: 136 - 7 - 3 + 1 = 127

时序划分:
  前80% (0-100号窗口):
    train_val_indices = [0..100] → shuffle → train(76) + val(25)
  后20% (101-126号窗口):
    test_indices = [101..126]  # 严格保序, 模拟真实预测
```

### 3.2 归一化器拟合 (防泄露)

```
仅用训练集涉及的天数拟合归一化器:
  case_scaler = CityLevelLogMinMaxScaler()
    fit(train_features[:,:,:3])
    → 对病例特征逐城市独立做 log1p → min-max

  other_scaler = CityLevelLogMinMaxScaler()
    fit(train_features[:,:,3:])
    → 对其他特征逐城市独立归一化

关键设计:
  - 静态特征(方差≈0)退回全局min-max, 保留空间差异
  - 验证集和测试集用训练集的scaler变换, 无信息泄露
```

### 3.3 数据集与DataLoader

```
train_ds = ImprovedTemporalDataset(features, adjs, ...,
                                    indices=train_indices,
                                    case_scaler, other_scaler)
val_ds, test_ds 同理

__getitem__ 返回:
  x:       (7, 24, 17)   归一化后的lookback窗口
  adj_seq: (3, 24, 24)   预测期的迁徙矩阵
  y_norm:  (3, 24, 3)    归一化后的目标 (儿童/成人/老年)
  y_raw:   (3, 24, 3)    原始值目标 (保留用于最终评估)

DataLoader:
  batch_size = min(8, len(train_ds))  # 通常=8
  train: shuffle=False, drop_last=True
  val:   shuffle=False, drop_last=False
  test:  shuffle=False, drop_last=False
```

### 3.4 接触矩阵

```
raw_contact = [[10.0, 4.5, 0.8],    # 儿童→[儿童,成人,老年]
               [2.8, 11.5, 1.2],    # 成人→[儿童,成人,老年]
               [0.6, 1.5, 3.5]]     # 老年→[儿童,成人,老年]

谱半径归一化: base_contact = raw_contact / ρ(raw_contact)
→ ρ ≈ 13.14 → base_contact 最大特征值=1.0
→ 解耦物理尺度与神经网络激活域
```

---

## 4. 阶段2A: 模型初始化

### 函数: `setup_models()` (L1436-1480)

### 7个模型的本质差异

所有Graphormer系列模型共用 `EnhancedFull_Graphormer_V3` 类，通过3个布尔开关控制消融：

```
                    use_tcn   enable_physics   use_kan
Full_KAN_Graphormer    ✓           ✓             ✓      ← 完全体
M_Graphormer_Baseline  ✓           ✓             ✗      ← KAN→MLP
NoPhysics_KAN_Gphormer ✓           ✗             ✓      ← 关物理
KAN_Only               ✗           ✓             ✓      ← 关图结构
GAT_Baseline           (GAT代替)   ✗             ✗      ← 图注意力基线
LSTM_Baseline          (纯LSTM)    ✗             ✗      ← 时序基线
MLP_Baseline           (纯MLP)     ✗             ✗      ← 最简单基线
```

### 模型配置速查

| 模型 | 参数量 | LR | patience | loss_type |
|------|--------|-----|----------|-----------|
| Full_KAN_Graphormer | 18,717 | 0.0005 | 50 | dynamic (课程学习) |
| M_Graphormer_Baseline | 15,974 | 0.0005 | 40 | dynamic |
| NoPhysics_KAN | ~18K | 0.0005 | 50 | standard (纯Huber) |
| KAN_Only | ~19K | 0.0005 | 40 | dynamic |
| GAT_Baseline | ~10K | 0.0005 | 30 | standard |
| LSTM_Baseline | ~3K | 0.001 | 40 | standard |
| MLP_Baseline | ~6K | 0.001 | 30 | standard |

---

## 5. 阶段2B: 核心模型前向传播

### 类: `EnhancedFull_Graphormer_V3.forward()` (L968-1033)

这是整个框架的核心。下面以 **Full_KAN_Graphormer** (完全体) 为例，逐步骤追踪形状变化。

### 输入

```
x:        (B, L=7, N=24, F=17)    归一化特征窗口, B≤8
adj_seq:  (B, 3, 24, 24)          预测期迁徙矩阵
dist_matrix: (B, 24, 24)          扩散距离 (由邻接矩阵计算)
time_step: 可选, 时间步标记
```

### 步骤1: 输入投影 + 中心度编码

```python
# 取最后一天的快照作为空间特征
x_last = x[:, -1, :, :]                 # (B, 24, 17)

# 线性投影到隐层空间
h = self.input_proj(x_last)             # Linear(17, 16) → (B, 24, 16)

# 中心度编码：防止度数信息被注意力淹没
adj_binary = (adj_cur > 1e-4).float()   # 二值化
in_deg = adj_binary.sum(dim=2)          # (B, 24) 每城入度
out_deg = adj_binary.sum(dim=1)         # (B, 24) 每城出度
h = h + in_degree_embed(in_deg)         # Embedding查表后相加
h = h + out_degree_embed(out_deg)       # (B, 24, 16)
```

**为什么用度数编码**: Graphormer原论文用度数向量加法，本文改为拼接+投影(`[xi‖z⁻‖z⁺]·W_proj`)，避免特征被度数对消。

### 步骤2: Gated TCN 时序特征

```python
# 将时序维度重排
x_seq = x.permute(0,2,1,3)              # (B, 24, 7, 17)
x_seq = x_seq.reshape(B*N, L, F)        # (B×24, 7, 17)
x_seq_proj = input_proj(x_seq)          # (B×24, 7, 16)
x_seq_proj = x_seq_proj.transpose(1,2)  # (B×24, 16, 7) ← Conv1d格式

# 两层扩张因果卷积 + 门控
tcn_out = tcn1(x_seq_proj)              # dilation=1 → (B×24, 16, 7)
tcn_out = tcn2(tcn_out)                 # dilation=2 → (B×24, 16, 7)
tcn_out = tcn_out + x_seq_proj          # 残差连接

# GatedTCN内部:
#   conv_out  = Conv1d(x)               # 特征通道
#   gate_out  = sigmoid(GateConv1d(x))  # 门控通道 (0~1)
#   output    = conv_out ⊙ gate_out     # 自适应噪声过滤

# 融合到时序最后一步
tcn_last = tcn_out[:, :, -1]            # (B×24, 16)
tcn_last = tcn_last.reshape(B, N, 16)   # (B, 24, 16)
h = h + tcn_last                        # (B, 24, 16)
```

**为什么用Gated TCN**: 并行计算 + 可控感受野 + 门控自适应过滤疫情数据噪声。

### 步骤3: Graphormer 空间建模

```python
layer_outputs = [h]  # Jumping Knowledge初始缓存

for i, layer in enumerate(graphormer_layers):
    # === 门控跳跃连接 ===
    if i == 0:
        h_in = h                              # (B, 24, 16)
    else:
        # gate × 当前 + (1-gate) × 历史均值
        gate = sigmoid(dense_gates[i])        # (1,)
        history_mean = mean(layer_outputs)    # (B, 24, 16)
        h_in = gate*h + (1-gate)*history_mean # (B, 24, 16)

    # === GraphormerLayer.forward() ===
    # 3.1 空间编码 (RBF)
    spatial_bias = RBFSpatialEncoding(dist_matrix)
    # dist_matrix: (B,24,24) → RBF展开 → MLP → (B, heads, 24, 24)
    # 高斯核: exp(-|d-c|²/w²), 16个均匀分布中心

    # 3.2 边编码 (动态邻接)
    edge_bias = Linear(adj_cur.unsqueeze(-1)) # (B, heads, 24, 24)
    # 实时感知封城等管制导致的流动变化

    # 3.3 注意力计算
    attn_mask = spatial_bias + edge_bias      # (B, heads, 24, 24)
    attn_mask = attn_mask.masked_fill(~topology_mask, -inf)
    # 只对有连边的城市对计算注意力
    x_attn, attn_weights = MultiheadAttention(x, x, x, mask=attn_mask)
    # 注意力分数 = Q·K^T/√dk - α·d_diff + ε_ij
    #                 缩放点积    扩散距离偏置  边编码

    # 3.4 残差 + 层归一化
    h_out = norm1(h_in + x_attn)              # (B, 24, 16)

    # 3.5 前馈网络 (KAN 或 MLP)
    ffn_out = FFN(h_out)                       # (B, 24, 16)
    # KAN模式: KAN(16→32) → LayerNorm → Dropout → KAN(32→16)
    # MLP模式: MLP(16→32) → LayerNorm → Dropout → MLP(32→16)
    h_out = norm2(h_out + Dropout(ffn_out))    # (B, 24, 16)

    h = h_out
    layer_outputs.append(h)
    gate = sigmoid(dense_gates[i])

# 最终融合（最后一层的门控）
h = gate*h + (1-gate)*mean(layer_outputs)     # (B, 24, 16)
```

**三个关键改进** (相对于标准Graphormer):
| 原始方案 | 本文改进 | 原因 |
|---------|---------|------|
| 向量加法中心度 | 拼接+投影 | 避免特征对消 |
| 最短路径距离 | 扩散距离 | 人口流动≠地理距离 |
| 静态边编码 | 动态边编码 `A(t)×W_E` | 实时感知封城 |

### 步骤4: 病例预测头 (NN直出)

```python
# 4.1 预测原始输出
cases_raw = case_head(h)                     # (B, 24, 9) = predict_window×3年龄组
cases_raw = cases_raw.view(B, N, 3, 3)      # (B, 24, 3, 3)
cases_raw = cases_raw.transpose(1, 2)        # (B, 3, 24, 3) ← 预测步×城市×年龄

# KAN模式 case_head:
#   KAN(16→8, grid_size=5) → KAN(8→9, grid_size=5, 小初始化)

# MLP模式 case_head:
#   Linear(16→8) → SiLU → Linear(8→9)  (逆Sigmoid初始化)

# 4.2 残差预测 (核心公式)
last_day = x[:, -1, :, 0:3].unsqueeze(1)   # (B, 1, 24, 3) 昨日病例
cases_pred_nn = clamp(last_day + tanh(cases_raw) × 0.5, 0.0, 1.2)
#                                               (B, 3, 24, 3)

# 关键设计:
#   - 预测增量而非绝对值: last_day + Δ
#   - tanh限制增量范围: [-0.5, +0.5] (归一化空间)
#   - clamp防溢出: [0.0, 1.2]
```

### 步骤5: 物理参数头

```python
h_phys = h  # (B, 24, 16)  注意: 不detach, 端到端可微

# 5.1 β (传播率)
beta_raw = beta_head(h_phys)                            # (B, 24, 3)
region_flag = x[:, -1, :, 10].unsqueeze(-1)             # (B, 24, 1) 南方=1
beta = sigmoid(beta_raw + softplus(region_boost)×南方) × 0.9 + 0.1
# → β ∈ [0.1, 1.0], 南方自动获得更高传染先验

# 5.2 接触系数
base_contact = sigmoid(contact_head(h_phys)) × 1.5 + 0.1  # (B, 24, 3) ∈ [0.1, 1.6]
lockdown = x[:, -1, :, 9].unsqueeze(-1)                   # (B, 24, 1) 封城强度
contact = base_contact × exp(-softplus(α) × lockdown)
# → 封城城市接触率指数衰减, α初始=0.5 (可学习)

# 5.3 γ (恢复率)
gamma = sigmoid(gamma_head(h_phys)) × 0.191 + 0.142       # (B, 24) ∈ [0.142, 0.333]
# → 病程 1/γ = 3~7天

# 5.4 i₀ (初始感染乘数)
i0_mult = sigmoid(i0_head(h_phys)) × 4.0 + 1.0            # (B, 24) ∈ [1.0, 5.0]
```

### 步骤6: 返回值

```python
return (
    cases_pred_nn,    # (B, 3, 24, 3)  NN直出预测 (归一化空间)
    beta,             # (B, 24, 3)     传播率
    gamma,            # (B, 24)        恢复率
    base_contact,     # (B, 24, 3)     基础接触系数 (无封城干预)
    contact,          # (B, 24, 3)     有效接触系数 (封城衰减后)
    i0_mult,          # (B, 24)        初始感染乘数
    attn_weights      # (B, 24, 24)    最后一层注意力权重
)
```

### 完整形状流总结

```
输入 x: (B, 7, 24, 17)
       │
       ├─ x[:, -1] ──► input_proj ──► (B, 24, 16)
       │                  │
       │    + 中心度编码 (in_degree + out_degree embedding)
       │    + TCN时序   (GatedTCN dilation=1,2)
       │                  │
       │                  ▼
       │              (B, 24, 16) ← 时序增强后的隐层
       │                  │
       │    + Graphormer × 2 layers
       │      ├ 空间编码: RBF(扩散距离) → (B, heads, 24, 24)
       │      ├ 边编码:   Linear(邻接)  → (B, heads, 24, 24)
       │      ├ 注意力:   MHA(Q,K,V, mask=空间+边) → (B, 24, 16)
       │      └ FFN:      KAN/MLP(16→32→16) → (B, 24, 16)
       │    + 门控跳跃连接 (JK-Net)
       │                  │
       │                  ▼
       │              (B, 24, 16) ← 最终共享隐层 h
       │                  │
       ├─ case_head ──────► tanh残差 ──► (B, 3, 24, 3)  病例预测
       │
       ├─ beta_head ─── sigmoid×0.9+0.1 ─► (B, 24, 3)  β 传播率
       ├─ gamma_head ── sigmoid×0.191+0.142 ► (B, 24)   γ 恢复率
       ├─ contact_head─ sigmoid×1.5+0.1 ──► (B, 24, 3)  C 基础接触
       │   × exp(-α × lockdown)            ► (B, 24, 3)  C 有效接触
       └─ i0_head ──── sigmoid×4.0+1.0 ──► (B, 24)     i0 初始感染
```

---

## 6. 阶段3: 训练循环

### 函数: `train_model()` (L1534-1768)

### 6.1 初始化

```python
# 优化器
optimizer = AdamW(model.parameters(), lr=模型特定值, weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6)

# 损失函数 (根据loss_type)
if loss_type == 'dynamic':
    criterion = CurriculumScientificLoss(total_epochs)  # 课程学习 + 多任务
else:
    criterion = None  # 纯 Huber 损失
```

### 6.2 训练阶段 (每个epoch)

```
for x, adj_seq, y_norm, y_raw in train_loader:
    x:       (B, 7, 24, 17)   → device
    adj_seq: (B, 3, 24, 24)   → device
    y_norm:  (B, 3, 24, 3)    → device

    # --- 前向传播 ---
    if has_physics:
        # 网络前向
        cases_pred_nn, beta, gamma, base_contact, contact, i0_mult, _ = model(x, adj_seq, dist)

        # SEIR 初始状态估计
        I0_log = cases_pred_nn中最后一天的病例对数
        I0_raw = expm1(I0_log) × i0_mult  # 原始病例 (B, 24, 3)
        E0_raw = I0_raw × 1.5             # 潜伏期估计
        S0 = pop - I0_raw - E0_raw        # 易感人群
        R0 = 0                             # 恢复人群

        # ODE 推演循环 (3个预测步)
        for t in range(3):
            S0, E0, I0, R0, incidence = age_structured_seir_step(
                S0, E0, I0, R0,
                beta[:, :, t],      # (B, 24)
                gamma,              # (B, 24)
                contact[:, :, t],   # (B, 24)
                adj_seq[:, t],      # (B, 24, 24)
                pop,                # (B, 24, 3)
                base_contact,       # (3, 3)
                mobility_rate=0.01
            )
            # → RK4, 2子步, dt=1
            # → incidence 积分潜伏期转化率
            # → log1p → 归一化 → clamp[0, 1.2]

        cases_ode_norm = stack(3个incidence)  # (B, 3, 24, 3)

        # 多任务损失
        loss_data = Huber(cases_pred_nn, y_norm)
        loss_physics = Huber(cases_ode_norm, y_norm)
        R0 = calculate_R0_NGM(beta_mean, gamma, base_contact_mean, adj_mean)
        loss_r0 = Huber(R0, 1.3)
        tv_loss = ||β[t+1]-β[t]||² + ||C[t+1]-C[t]||²
        spline_loss = L2(KAN样条权重) + 平滑差分

        → curriculum × (0.05×loss_physics + 0.005×loss_r0
                        + 0.01×tv_loss + 0.001×spline_loss)
        → total_loss = loss_data + 加权和

    else:
        cases_pred_nn, _, _, _, _, _, _ = model(x, adj_seq, dist)
        loss = Huber(cases_pred_nn, y_norm)

    # --- 反向传播 ---
    loss.backward()
    clip_grad_value_(model.parameters(), 1.0)  # 梯度裁剪（值裁剪，非范数裁剪）
    optimizer.step()
```

### 6.3 SEIR ODE 积分 (核心)

```
age_structured_seir_step(S, E, I, R, β, γ, contact, mig, pop, base_contact):
  RK4 with 2 sub-steps, dt=1.0

  每个子步:
    dS/dt = μ·pop - μ·S - β·contact·S·I/N + Σ(mig·S_in - S_out)
    dE/dt = β·contact·S·I/N - (μ+σ)·E + Σ(mig·E_in - E_out)
    dI/dt = σ·E - (μ+λ_dis)·I - γ·I + Σ(mig·I_in - I_out)
    dR/dt = γ·I - μ·R + Σ(mig·R_in - R_out)

    incidence = σ·E  (新发报告病例 = 潜伏期→感染期转化)

  常量:
    σ = 1/3          (平均潜伏期3天)
    μ = 1/(70×365)   (背景死亡率)
    λ_dis = 0.00008  (疾病致死率)

  迁移矩阵耦合:
    effective_mig = (1-0.01)×I + 0.01×M  (99%本地+1%跨城)
```

### 6.4 损失函数详解 `CurriculumScientificLoss`

| 损失项 | 公式 | 基础权重 | 动态调整 |
|--------|------|---------|---------|
| data_loss | Huber(NN预测, 真实) | 1.0 | — |
| physics_loss | Huber(ODE推演, 真实) | 0.05 | × curriculum × scale_physics |
| r0_loss | Huber(NGM谱半径, 1.3) | 0.005 | × curriculum × scale_r0 |
| tv_loss | ‖β[t+1]-β[t]‖² + ‖C[t+1]-C[t]‖² | 0.01 | × curriculum |
| spline_loss | L2(β/γ/C/i0头的spline_weight) + 差分平滑 | 0.001 | × curriculum |

```
curriculum_factor = min(1.0, epoch / 20)  # 前20轮线性0→1

自适应缩放 (防止物理项压垮数据项):
  scale_physics = (loss_data / loss_physics).clamp(max=1.0)
  scale_r0 = (loss_data / loss_r0).clamp(max=1.0)

最终:
  total = loss_data
        + 0.05 × curriculum × scale_physics × loss_physics
        + 0.005 × curriculum × scale_r0 × loss_r0
        + 0.01 × curriculum × tv_loss
        + 0.001 × curriculum × spline_loss
```

### 6.5 验证阶段 (每个epoch)

```
with no_grad:
  前向传播 (同训练)
  但验证损失只用 NN 预测! (不用ODE)
    → 原因: ODE刚性约束在验证集可能比NN差,
      用ODE做验证损失会欺骗LR Scheduler导致过早死亡

  pred_raw = expm1(pred_norm × range + min)  # 逆归一化
  val_loss = Huber(pred_norm, y_norm)
  val_R² = r2_score(all_pred, all_target)

→ scheduler.step(val_loss)
→ 早停: patience次无改善则break
→ 保存最佳模型: {model_name}_best.pth
```

---

## 7. 阶段4: 评估与可视化

### 7.1 评估流程 `evaluate_model()` (L1770-1980)

```
with no_grad:
  for batch in test_loader:
    → 前向传播 → cases_pred_nn, β, γ, contact, etc.
    → cases_pred_raw = expm1(逆归一化)  # 还原为原始病例数

    → 计算 Rt (简化公式, 逐城市):
      prob_survive_E = σ / (σ + μ + outflow_i)       # E仓室存活概率
      duration_I = 1 / (γ + μ + λ_dis + outflow_i)   # I仓室平均停留
      Rt_i = (β_i × contact_i) × prob_survive_E × duration_I

    → DataCollector 收集:
      predictions, targets, betas, gammas, contacts, Rt, attn_weights, residuals

→ 全局指标:
  R² = r2_score(全部预测展平, 全部真实展平)
  RMSE = √(mean((pred-target)²))
  MAE = mean(|pred-target|)

→ 空间泛化分析:
  一线城市(北上广深) R² vs 二线+其他城市 R²

→ 流行病学参数统计:
  β: 均值±标准差 (流感合理区间: [0.10, 0.60])
  γ: 均值±标准差 (流感合理区间: [0.15, 0.35])
  contact: 均值±标准差
  Rt: 均值±标准差 + 有效性比例 (Rt∈[1.0, 2.0]的比例)

→ 残差诊断:
  均值, 偏度, 峰度, Durbin-Watson统计量
```

### 7.2 全时段预测 `predict_full_timeseries_with_params()` (L2038-2179)

```
滑动窗口遍历全部136天:
  for t in 0..126:
    x_window = features[t:t+7]     # lookback窗口
    → model(x_window, adj_window) → pred_norm, β, γ, contact
    → 逆归一化 → pred_raw
    → 累加 pred_sum[day] += pred_raw  (重叠窗口取平均)
    → 累加 β/contact/Rt 同理

输出:
  true_total:   (136, 24)   全时段真实病例
  pred_total:   (136, 24)   全时段预测病例
  beta_full:    (136, 24)   全时段β
  rt_full:      (136, 24)   全时段Rt
  contact_full: (136, 24)   全时段有效接触系数
```

### 7.3 封城效应分析

```python
# 以封城日 day 83 为界
pre_lockdown  = 参数[:83].mean()
post_lockdown = 参数[83:].mean()

# 典型案例 → 武汉:
#   β:     0.427 → 0.418  (不变 — 病毒生物属性)
#   contact: 0.567 → 0.276  (↓51% — 封城直接效应)
#   Rt:    1.315 → 0.668  (↓49% — 跨过1.0阈值, 疫情受控)

# 对照 → 青岛(未封城):
#   所有参数几乎不变
```

### 7.4 可视化清单 (20+图表)

```
visualizations/
├── architecture/          # 参数分布直方图 + 层结构图 + KAN权重可视化
├── training/              # 训练/验证loss曲线 + LR变化 + 每epoch耗时
├── prediction/            # 预测vs真实散点图 + 时序图 + 城市级热力图
├── parameters/            # β/Rt/contact分析图 + KAN样条曲线可视化
├── attention/             # 24×24注意力热力图 (识别传播关键路径)
├── residuals/             # 残差分布 + Q-Q图 (正态性检验)
├── error/                 # 残差自相关图 (Durbin-Watson诊断)
├── evolution/             # β/Rt/contact的24城×时间热力图
├── correlation/           # 参数-城市特征相关性散点图
└── comparison/            # 7模型消融性能对比 + 封城参数对比
    ├── performance_comparison.png     # R²/RMSE/MAE柱状图
    ├── performance_radar.png          # 雷达图
    ├── training_efficiency.png        # 训练效率对比
    ├── parameter_consistency.png      # 参数一致性
    └── lockdown_params_comparison.png # 武汉vs青岛封城对比 ✨
```

---

## 8. 附录: 关键公式与常量

### 8.1 SEIR 动力学方程

```
dS_i/dt = μ·N_i - β_i(t)·(S_i·I_i)/N_i - μ·S_i + Σ(m_ji·S_j - m_ij·S_i)
dE_i/dt = β_i(t)·(S_i·I_i)/N_i - (σ+μ)·E_i + Σ(m_ji·E_j - m_ij·E_i)
dI_i/dt = σ·E_i - (γ_i(t)+μ+λ_dis)·I_i + Σ(m_ji·I_j - m_ij·I_i)
dR_i/dt = γ_i(t)·I_i - μ·R_i + Σ(m_ji·R_j - m_ij·R_i)
```

### 8.2 Rt 简化公式 (评估用)

```
Rt_i = β_i × contact_i × [σ/(σ+μ+outflow_i)] × [1/(γ_i+μ+λ_dis+outflow_i)]

其中 outflow_i = 0.01 × Σ_j M_ij  (城市i的总迁出率)
```

### 8.3 NGM R₀ (损失函数用)

```
F = diag(β × contact_base)       # 新感染矩阵
V_E = diag(σ+μ+outflow) - M^T   # E仓室转移
V_I = diag(γ+μ+λ_dis+outflow) - M^T  # I仓室转移
K = F × V_I⁻¹ × diag(σ) × V_E⁻¹
R₀ = ρ(K) = max|eig(K)|           # 谱半径
```

### 8.4 KAN 激活函数

```
φ(x) = w_b × SiLU(x) + w_s × Σ c_i × B_i(x)
       └─ 基函数分支 ─┘   └── B样条可学习分支 ──┘

B样条: 局部支撑, 调整c_i只影响局部区间
数值稳定: x_clamped = 3.0 × tanh(x/3.0)  # 软截断到[-3,3]
```

### 8.5 重点代码行号索引

| 功能 | 行号 | 函数/类名 |
|------|------|-----------|
| 数据加载 | L4597 | `load_raw_data_only()` |
| 时序增强 | L337 | `TemporalFeatureEnhancer.add_temporal_features()` |
| CityLevelLogMinMaxScaler | L195 | 逐城市独立归一化 |
| GatedTCN | L613 | 门控时序卷积 |
| RobustKANLinear | L657 | B样条KAN层 |
| RBFSpatialEncoding | L712 | 扩散距离RBF编码 |
| GraphormerLayer | L832 | 图注意力+FFN |
| EnhancedFull_Graphormer_V3 | L887 | **核心模型前向传播** |
| SEIR导数 | L737 | `compute_derivatives_seir()` |
| RK4积分 | L772 | `age_structured_seir_step()` |
| NGM R0 | L1233 | `calculate_R0_NGM()` |
| CurriculumScientificLoss | L1273 | 课程学习多任务损失 |
| ImprovedAblationStudyManager | L1356 | 训练/评估管理器 |
| 训练循环 | L1534 | `train_model()` |
| 评估流程 | L1770 | `evaluate_model()` |
| 全时段预测+参数 | L2038 | `predict_full_timeseries_with_params()` |
| 主函数 | L4706 | `main()` |

---

*文档生成日期: 2026-06-18 | 基于 exp_chronological.py v9 (5058行)*
