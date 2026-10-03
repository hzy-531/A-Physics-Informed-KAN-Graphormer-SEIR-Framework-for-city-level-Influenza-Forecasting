# Rt 有效再生数分析方法

## 概述

模型通过物理解码头同时输出 **β (传播率)**、**γ (恢复率)**、**contact (接触系数)**，再从这些参数计算 **Rt (有效再生数)** 和 **R₀ (基本再生数/NGM谱半径)**。

---

## 一、参数估计（模型输出）

### 1.1 解码头

四个参数头共享 backbone 隐层输出 `h_phys`，各为 2 层 KAN/MLP：

| 参数 | 后处理公式 | 值域 | 流行病学含义 |
|------|-----------|------|------------|
| β (传播率) | `sigmoid(β_raw + region_boost × 南方) × 0.9 + 0.1` | [0.1, 1.0] | 单次接触感染概率 |
| γ (恢复率) | `sigmoid(γ_raw) × 0.191 + 0.142` | [0.142, 0.333] | 1/γ = 3~7天病程 |
| contact (接触) | `sigmoid(C_raw) × 1.5 + 0.1` → `× exp(-α × lockdown)` | [0.1, 1.6] | 接触强度修正 |
| i₀ (初始感染) | `sigmoid(i0_raw) × 4.0 + 1.0` | [1.0, 5.0] | SEIR 初始 I 乘数 |

### 1.2 封城衰减

```
contact_effective = contact_base × exp(-softplus(α) × lockdown_strength)
```

- α 初始化为 0.5（可学习参数）
- lockdown_strength: 武汉=0.8, 石家庄=0.6, 哈尔滨=0.5, 其他=0

---

## 二、Rt 计算（简化公式）

### 2.1 评估时 Rt

```python
mu = 1.0 / (70 × 365)           # 背景死亡率
lambda_dis = 0.00008             # 疾病致死率
sigma = 1.0 / 3.0                # 潜伏期转化率 (3天潜伏期)

effective_mig = 0.01 × M         # 有效迁移矩阵 (M = 日迁徙矩阵)
outflow = Σ_j M_ij               # 每城市总迁出率

# 三段式 Rt:
prob_survive_E = sigma / (sigma + mu + outflow)        # 潜伏期内不被移除的概率
duration_I     = 1.0 / (gamma + mu + lambda_dis + outflow)  # 感染期平均时长
Rt_local       = (β × contact) × prob_survive_E × duration_I
```

**直觉解读**：
- `β × contact` = 有效传播率（单感染者每日所致新感染数）
- `prob_survive_E` = 进入潜伏期后存活至发病的概率
- `duration_I` = 感染期患者在被移除（恢复/死亡/迁出）前的平均停留天数

### 2.2 为什么用简化公式而非 NGM？

全矩阵 NGM 涉及每时间步的 `24×24` 矩阵求逆，在评估循环中代价高昂。简化公式逐城市计算，在迁出率低的城市与 NGM 结果高度一致。当 `outflow → 0` 时：

```
Rt → β × contact / gamma  （经典 SIR 的 R₀ 公式）
```

---

## 三、R₀ 计算（NGM 谱半径，仅用于损失函数）

`calculate_R0_NGM()` 在 `physics.py` 中实现，遵循 Diekmann 下一代矩阵方法：

### 3.1 构建传输矩阵 F

```
F = diag(β × contact)        # 新感染矩阵 (I→E)
```

### 3.2 构建转移矩阵 V

```
V_E = diag(sigma + mu + outflow_i) - M^T     # E 仓室流出
V_I = diag(gamma + mu + lambda_dis + outflow_i) - M^T  # I 仓室流出
```

### 3.3 NGM 与 R₀

```
K  = F × V_I⁻¹ × diag(sigma) × V_E⁻¹
R₀ = ρ(K) = max(|eig(K)|)          # 谱半径
```

### 3.4 R₀ 先验约束

```python
loss_r0 = Huber(R₀, 1.3, δ=0.5)
weight_r0 = 0.005 × curriculum × (loss_data / loss_r0).clamp(max=1.0)
```

将 NGM 谱半径 R₀ 拉向先验值 1.3（中等传播强度），防止参数估计发散。

---

## 四、Rt 爆发概率分析

### 4.1 阈值判断

| Rt 范围 | 流行病学状态 |
|---------|------------|
| Rt < 0.5 | 疫情消退（可能低估） |
| 0.5 ≤ Rt < 1.0 | 疫情受控、逐渐缩小 |
| 1.0 ≤ Rt ≤ 2.0 | 局部爆发、持续传播 |
| Rt > 2.0 | 大规模爆发（可能高估） |

### 4.2 有效性比例

```python
valid_ratio = mean((Rt >= 0.5) & (Rt <= 2.0))
```

论文报告 93.06% 的 Rt 估计值落在 [0.5, 2.0] 内。

### 4.3 封城前后对比

按封城日 (day 83) 切分，对比封城前后参数均值：

```python
pre  = mean(参数[:83])
post = mean(参数[83:])
```

典型案例（武汉）：
- β: 0.427 → 0.418（基本不变，病毒生物属性）
- contact: 0.567 → 0.276（↓51%，封城直接效应）
- Rt: 1.315 → 0.668（↓49%，跨过 1.0 阈值）

### 4.4 南北差异

南方 β=0.561, Rt=1.168 → 疫情更活跃
北方 β=0.399, Rt=0.825 → 传播受抑制

供暖标识与 β 强负相关 (r=-0.633)，干燥环境可能抑制传播。

---

## 五、可视化诊断

### 5.1 Rt 分析图 (`_plot_r0_analysis`)
- **时间序列**: 平均 Rt ± 标准差，1.0 阈值红线
- **分布直方图**: 全时段 Rt 值分布，标记均值
- **状态饼图**: Rt<1.0（受控）vs Rt≥1.0（爆发）比例

### 5.2 封城参数对比 (`_plot_lockdown_params_comparison`)
- 双轴图：左轴 β/contact，右轴 Rt
- 封城竖线标记 + 前后均值虚线
- 统计标注：变化百分比和显著性

### 5.3 参数演化热力图 (`plot_parameter_evolution`)
- 24 城 × 时间的 β/Rt/contact 热力图
- 城市按纬度/区域排序，观察南北梯度

---

## 六、代码位置速查

| 想知道什么 | 去哪里 |
|-----------|--------|
| SEIR ODE 方程 | `exp_lib/physics.py` → `compute_derivatives_seir()` |
| RK4 求解器 | `exp_lib/physics.py` → `age_structured_seir_step()` |
| NGM R₀ 谱半径 | `exp_lib/physics.py` → `calculate_R0_NGM()` |
| 参数头后处理 | `exp_lib/models.py` → `EnhancedFull_Graphormer_V3.forward()` |
| 评估时 Rt 计算 | `exp_lib/trainer.py` → `evaluate_model()` L456-468 |
| 全时段 Rt 计算 | `exp_lib/trainer.py` → `predict_full_timeseries_with_params()` L689-700 |
| R₀ 损失正则化 | `exp_lib/loss.py` → `CurriculumScientificLoss.forward()` |
| Rt 可视化 | `exp_lib/visualizer.py` → `_plot_r0_analysis()`, `_plot_lockdown_params_comparison()` |
| 封城定义 | `exp_lib/config.py` → `LOCKDOWN_INFO` |
