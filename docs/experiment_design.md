# 实验设计与方法详解

## 数据划分

### 时序划分 (当前方案)

- 滑动窗口 lookback=7, predict=3 → 总样本 136-7-3+1=127
- 训练 76 (60%) + 验证 25 (20%) + 测试 26 (20%)
- **方案**: 前80%窗口(0-100)内 train+val 随机打乱，后20%窗口(101-126)严格保序做测试
- 归一化器 `CityLevelLogMinMaxScaler` 仅用训练集天数拟合，再全局应用
- 代价：train↔val 有窗口重叠泄露（相邻窗口共享 6/7 天输入）
- 原因：136天仅一个疫情周期，纯时序划分会导致 train/val/test 处于不同阶段

### 随机划分 (备用，exp_random.py)

- 先按滑动窗口划分好 127 个样本 → 全局打乱 → 70/15/15 切分
- 造成时间泄露：day 120 在训练集，day 10 可能在测试集
- 论文将其作为对照实验——给出 IID 假设下的理论上限

### 验证集 vs 测试集

| 属性 | 验证集 | 测试集 |
|------|--------|--------|
| 作用 | 过程监督：每epoch看，决定早停和LR调度 | 最终裁判：训练结束后只用一次 |
| 权重更新 | ❌ | ❌ |
| 使用次数 | 几十次 | 一次 |
| 数据分布 | 与训练同时间段（前80%） | 严格未来数据（后20%） |

## 封城效应

### 封城城市定义 (`exp_lib/config.py` LOCKDOWN_INFO)

| 城市 | 时间段 | 强度 |
|------|--------|------|
| 武汉 | 1/23→4/08 (76天) | 0.8 |
| 石家庄 | 1/24→2/09 (16天) | 0.6 |
| 哈尔滨 | 2/4→3/4 (28天) | 0.5 |
| 北京及其他 | start=None | - |

判断标准：`LOCKDOWN_INFO[city]['start'] is not None`

### 封城效应归因

- β (传播率) 封城前后基本持平 → 病毒生物属性，封城改不了
- C (接触系数) 武汉封城后腰斩 (0.57→0.28)，青岛不变 → 模型正确归因
- Rt 武汉从 1.32→0.67 跨过 1.0 阈值，青岛无变化

## 物理 ODE 双路架构

```
输入 x → backbone (Graphormer+TCN+KAN) → h (共享隐层)
                                            │
              ┌─────────────────────────────┤
              ▼                             ▼
        case_head                      物理参数头
        (黑箱预测)                  ┌── beta_head → β [0.1,1.0]
              │                    ├── gamma_head → γ [0.14,0.33]
              ▼                    ├── contact_head → C [0.1,1.6]
        NN预测(3天)                │    × exp(-α×封城强度)
              │                    └── i0_head → i0_mult [1,5]
              │                             │
              │                             ▼
              │                  SEIR ODE (RK4, 2子步)
              │                  S→E→I→R 推演3天
              │                             │
              ▼                             ▼
        loss_data                   loss_physics
        权重 1.0                     权重 0.02-0.05
              └──────────┬──────────────────┘
                         ▼
                    total_loss
```

### 参数公式

```python
beta    = sigmoid(beta_head(h) + region_boost × 南方) × 0.9 + 0.1
gamma   = sigmoid(gamma_head(h)) × (0.333-0.142) + 0.142  # 病程3~7天
contact = sigmoid(contact_head(h)) × 1.5 + 0.1
contact = contact × exp(-softplus(α) × 封城强度)           # 封城衰减
i0_mult = sigmoid(i0_head(h)) × 4.0 + 1.0
```

### 损失函数 (CurriculumScientificLoss)

| 项 | 公式 | 权重 |
|------|------|------|
| data_loss | Huber(NN预测, 真实) | 1.0 |
| physics_loss | Huber(ODE推演, 真实) | 0.05 × curriculum × scale |
| r0_loss | Huber(R₀, 1.3) | 0.005 × curriculum |
| tv_loss | β/contact时序平滑 | 0.01 × curriculum |
| spline_loss | KAN样条L2 | 0.001 × curriculum |

- curriculum: 前20 epoch 从0→1线性增长
- 验证时仅用NN预测计算val loss，不用ODE推演
- 预测3天但评估只取第1天 (`predictions[:, 0, :, :]`)

## 时间范围与数据

- 2019-11-01 ~ 2020-03-15（136天），含完整流感周期
- 特征维度：基础11维 + 时序增强6维 = 最终17维
- 迁移矩阵：136天×24×24，行和=1.0，SEIR中用 `0.99×I + 0.01×迁移矩阵`
- 详细数据说明见 `data/合并数据/statement.md`
