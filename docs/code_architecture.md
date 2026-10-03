# 代码架构详解

## 目录结构

```
code/code/
├── exp_lib/                          # 共享库 (从 exp_chronological.py 拆分)
│   ├── __init__.py                   # 统一导出
│   ├── config.py                     # 全局配置、路径、城市列表、随机种子
│   ├── layers.py                     # RobustKANLinear, StandardMLP, GatedTCN, GraphormerLayer, GAT, 空间编码
│   ├── physics.py                    # SEIR ODE、RK4 求解器、NGM R₀ 计算
│   ├── data.py                       # DataCollector、归一化器、时序增强、数据集、数据加载
│   ├── models.py                     # 全部 7 个消融模型 + MODEL_DEFINITIONS
│   ├── loss.py                       # CurriculumScientificLoss
│   ├── trainer.py                    # ImprovedAblationStudyManager
│   └── visualizer.py                 # ImprovedActualDataVisualizer
├── experiments/
│   ├── exp_chronological.py          # 原版主程序 (5058行，保留不动)
│   ├── exp_random.py                 # 随机划分（备用）
│   ├── run_single.py                 # 【调试】单模型×1种子
│   ├── run_kan_vs_mlp.py             # 【对照】KAN vs MLP 对比实验
│   ├── run_ablation.py               # 【调试】7模型×1种子
│   ├── run_3seeds.py                 # 【正式】7模型×3种子
│   ├── plot_results.py               # 【画图】读取pkl，不训练
│   └── plot_city_timeseries.py       # 【画图】24城病例时间序列
├── preprocessing/                    # 数据预处理脚本
└── analysis/                         # 可解释性分析
```

## 模块依赖关系

```
config.py  ←── 所有模块 (无内部依赖)
layers.py  ←── config
physics.py ←── 无 (纯数学)
data.py    ←── config
loss.py    ←── physics
models.py  ←── layers + physics
trainer.py ←── config + models + data + loss + physics
visualizer.py ←── config + layers
```

## 7 模型定义

全部在 `exp_lib/models.py`，通过 `MODEL_DEFINITIONS` 字典注册。

| 模型类 | 图结构 | KAN | 物理 | 用途 |
|--------|--------|-----|------|------|
| `EnhancedMLP_Baseline_V3` | - | - | - | 纯 MLP 基线 |
| `EnhancedLSTM_Baseline_V3` | - | - | - | LSTM 基线 |
| `EnhancedGAT_Baseline_V3` | GAT | - | - | 图注意力基线 |
| `M_Graphormer_Baseline` | Graphormer | - | Y | Graphormer+MLP+物理 |
| `EnhancedKAN_Only_V3` | - | Y | Y | 纯 KAN，无图结构 |
| `NoPhysics_KAN_Graphormer` | Graphormer | Y | - | KAN+图，关物理 |
| `EnhancedFull_Graphormer_V3` | Graphormer | Y | Y | 完全体 |

## 核心类/函数索引

| 要找什么 | 去哪里 |
|----------|--------|
| 改城市列表、封城参数 | `exp_lib/config.py` |
| 改 KAN 实现 | `exp_lib/layers.py` → `RobustKANLinear` |
| 改注意力机制 | `exp_lib/layers.py` → `GraphormerLayer`, `ImprovedGATLayer` |
| 改 SEIR 微分方程 | `exp_lib/physics.py` → `compute_derivatives_seir` |
| 改数据加载/归一化 | `exp_lib/data.py` → `load_raw_data_only`, `CityLevelLogMinMaxScaler` |
| 改模型架构 | `exp_lib/models.py` → 对应的模型类 |
| 改损失函数 | `exp_lib/loss.py` → `CurriculumScientificLoss` |
| 改训练流程 | `exp_lib/trainer.py` → `ImprovedAblationStudyManager` |
| 改图表样式 | `exp_lib/visualizer.py` → 对应的 `plot_*` / `_plot_*` 方法 |

## 关键设计决策

- **数据划分**: `run_ablation.py` `main()` 中实现 — 前80%窗口(train+val)随机打乱，后20%严格保序做测试
- **归一化器**: `CityLevelLogMinMaxScaler` 仅用训练集天数拟合，再全局应用
- **统一入口**: `run_all_experiments()` 内部调用 `setup_models()` + `train_model()` + `evaluate_model()` + 全时段预测
- **pkl 保存**: 每个模型单独保存 `{model_name}_result.pkl`，全时段参数额外保存 `full_pred.pkl`
- **向后兼容**: `plot_results.py` 将旧类注册到 `__main__` 以加载旧 pkl
- **KAN vs MLP 对照**: `GraphormerLayer` 内置 `use_kan` 开关 — True 用 RobustKANLinear (B-spline)，False 用 StandardMLP (Linear→SiLU→Linear)；实验侧 run_kan_vs_mlp.py 只对比 Full_KAN_Graphormer vs M_Graphormer_Baseline，其余架构完全一致
- **已有 bug 修复**: `M_Graphormer_Baseline` 和 `NoPhysics_KAN_Graphormer` 的 `__init__` 加 `**kwargs` 以兼容 trainer 传入的 `use_tcn/enable_physics/use_kan` 参数
