# 实验结果参考

## KAN vs MLP 对照实验 (150轮×1种子, 2026-06-16)

**配对**: Full_KAN_Graphormer vs M_Graphormer_Baseline
**变量**: 唯一差异是激活层 — RobustKANLinear (B-spline) vs StandardMLP (Linear+SiLU)
**架构**: 完全相同 — Graphormer + TCN + 物理ODE

| 指标 | KAN | MLP | Δ |
|------|-----|-----|-----|
| Test R² | **0.7414** | 0.7285 | **+0.0129** |
| RMSE | **3.5204** | 3.6073 | **−2.4%** |
| MAE | **1.9722** | 2.0291 | **−2.8%** |
| Val R² (best) | 0.9421 | **0.9426** | −0.0005 |
| 泛化落差 (Val−Test) | **0.2007** | 0.2141 | **−6.3%** |
| 一线城市 R² | **0.6869** | 0.6428 | **+0.0441** |
| 其他城市 R² | **0.7264** | 0.7226 | +0.0038 |
| 参数量 | 18,717 | **15,974** | +17% |
| β (传播率) | **0.424** | 0.519 | 文献值 0.46 |
| γ (恢复率) | **0.222** | 0.260 | — |
| Rt | 1.011 | 0.949 | — |

### 关键结论

1. **KAN 泛化更好**: MLP 验证集 R² 反而更高，但测试集输 0.0129，过拟合更严重
2. **一线城市差距最大**: 北京/上海/广州等超大城市 +0.0441 R²，B-spline 对复杂模式优势明显
3. **物理参数更合理**: KAN β=0.424 接近文献值 0.46 (PMC 2020)，MLP β=0.519 偏高 13%
4. **文献支撑**: 封城后中国 Rt 降至 0.59–0.67 (Zhang 2020, PLOS ONE)；β 典型值 0.2–0.8 day⁻¹

详见 [run_kan_vs_mlp.py](../code/code/experiments/run_kan_vs_mlp.py)

---

## 7 模型消融结果 (136天×3种子均值)

| 模型 | 图结构 | KAN | 物理 | R²(均值±σ) |
|------|--------|-----|------|-------------|
| M_Graphormer_Baseline | Graphormer | - | Y | **0.7547±0.0013** |
| GAT_Baseline | GAT | - | - | 0.7515±0.0071 |
| NoPhysics_KAN_Graphormer | Graphormer | Y | - | 0.7473±0.0037 |
| LSTM_Baseline | - | - | - | 0.7462±0.0040 |
| KAN_Only | - | Y | Y | 0.7454±0.0056 |
| Full_KAN_Graphormer | Graphormer | Y | Y | 0.7449±0.0147 |
| MLP_Baseline | - | - | - | 0.7374±0.0032 |

## 24城测试集 R² 排名 (Full_KAN_Graphormer)

| 最好 | R² | 最差 | R² |
|------|-----|------|-----|
| 长春 | 0.786 | 北京 | 0.320 |
| 深圳 | 0.780 | 杭州 | 0.316 |
| 广州 | 0.747 | 天津 | −0.448 |

## 武汉 vs 青岛 封城前后参数

| 参数 | 城市 | 封城前 | 封城后 | 变化 |
|------|------|--------|--------|------|
| β | 武汉 | 0.427 | 0.418 | −2% |
| β | 青岛 | 0.300 | 0.281 | −6% |
| C | 武汉 | 0.567 | 0.276 | **↓51%** |
| C | 青岛 | 0.566 | 0.544 | −4% |
| Rt | 武汉 | 1.315 | 0.668 | **↓49%** |
| Rt | 青岛 | 0.920 | 0.910 | −1% |

## 输出图表清单

运行后 `visualizations/` 目录结构：

```
visualizations/
├── architecture/       # 模型参数分布、层结构
├── training/           # 训练曲线 (loss/LR/time)
├── prediction/         # 散点图、时序图、城市级热力图
├── parameters/         # β/Rt/Contact 分析、KAN spline 曲线
├── attention/          # 注意力热力图
├── residuals/          # 残差分布、Q-Q图
├── error/              # 误差诊断 (自相关等)
├── evolution/          # 参数演化热力图
├── correlation/        # 参数-城市特征相关
└── comparison/         # 消融对比图、封城参数对比
    ├── performance_comparison.png
    ├── performance_radar.png
    ├── training_efficiency.png
    ├── parameter_consistency.png
    └── lockdown_params_comparison.png  ← 武汉vs青岛封城对比
```
