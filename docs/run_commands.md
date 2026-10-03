# 运行命令 & 参数参考

## 环境

**Python**: `E:\Claude code\KAN+\.venv\Scripts\python.exe`（torch 2.12.1+cpu, pandas 3.0.3）
**工作目录**: `E:\Claude code\KAN+\code\code`
**必须加**: `PYTHONIOENCODING=utf-8`（否则 emoji 导致 GBK 崩溃）

## 命令速查

```bash
cd "E:\Claude code\KAN+\code\code"

# 改代码后快速验证 (1模型 × 1种子，~2-5min)
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/run_single.py --epochs 20

# KAN vs MLP 对照实验 (2模型 × 1种子，~3min)
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/run_kan_vs_mlp.py --epochs 30

# 完整消融 (7模型 × 1种子，~20min)
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/run_ablation.py

# 正式实验 (7模型 × 3种子，~1h)
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/run_3seeds.py

# 画图 (读已有pkl，秒级，自动选最新结果)
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/plot_results.py
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/plot_results.py --dir <指定目录>
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/plot_results.py --dirs <目录1> <目录2> --summary

# 24城新增病例时间序列图
PYTHONIOENCODING=utf-8 E:\Claude code\KAN+\.venv\Scripts\python.exe experiments/plot_city_timeseries.py
```

---

## run_single.py — 单模型调试

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `Full_KAN_Graphormer` | 模型名（可选：MLP_Baseline, LSTM_Baseline, GAT_Baseline, M_Graphormer_Baseline, KAN_Only, NoPhysics_KAN_Graphormer） |
| `--seed` | int | 5780 | 随机种子 |
| `--epochs` | int | 模型预设值 | 训练轮数 |
| `--output` | str | 自动创建 | 输出目录 |
| `--no-plot` | flag | False | 跳过绘图 |

**示例：**
```bash
# 快速验证（20 epoch）
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_single.py --epochs 20

# 指定模型和种子
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_single.py --model M_Graphormer_Baseline --seed 42 --epochs 50

# 只训练不画图
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_single.py --no-plot
```

---

## run_kan_vs_mlp.py — KAN vs MLP 对照

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seed` | int | 5780 | 随机种子 |
| `--epochs` | int | 150 | 训练轮数 |
| `--output` | str | 自动创建 | 输出目录 |

运行 `Full_KAN_Graphormer` vs `M_Graphormer_Baseline`，唯一差异是激活层（B-spline vs MLP SiLU）。
输出 `kan_vs_mlp_summary.json` 含 R² 增量、RMSE/MAE 改进百分比、参数量对比。

**示例：**
```bash
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_kan_vs_mlp.py --epochs 150
```

---

## run_ablation.py — 7模型消融 (1种子)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seed` | int | 5780 | 随机种子 |
| `--epochs` | int | 模型预设值 | 训练轮数 |
| `--output` | str | 自动创建 | 输出目录 |

运行全部 7 个模型，生成完整可视化（训练曲线、预测散点、参数分析、注意力热力图、消融对比、封城参数对比）。
输出 `summary.json` 含所有模型 R²/RMSE/MAE。

**示例：**
```bash
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_ablation.py --epochs 100
```

---

## run_3seeds.py — 7模型消融 (3种子)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seeds` | str | `368,1037,5780` | 逗号分隔种子列表 |
| `--epochs` | int | 模型预设值 | 训练轮数 |

对每个种子独立调用 `run_ablation.main()`。单个种子失败不中断后续。用 `plot_results.py --dirs --summary` 汇总。

**示例：**
```bash
# 默认3种子
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_3seeds.py

# 自定义种子
PYTHONIOENCODING=utf-8 ...python.exe experiments/run_3seeds.py --seeds 100,200,300 --epochs 80
```

---

## plot_results.py — 纯画图

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dir` | str | 自动最新 | 单个结果目录 |
| `--dirs` | str[] | None | 多个结果目录（配合 --summary） |
| `--model` | str | 全部 | 只画指定模型 |
| `--comparison-only` | flag | False | 只画消融对比图 |
| `--summary` | flag | False | 多种子汇总（需 --dirs） |

**示例：**
```bash
# 自动选最新结果画全部图
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_results.py

# 指定目录
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_results.py --dir E:\Claude code\KAN+\result\ablation_study_v9_20260618_150157

# 只画消融对比图
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_results.py --comparison-only

# 多种子汇总
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_results.py --dirs <目录1> <目录2> <目录3> --summary
```

---

## plot_city_timeseries.py — 24城时序图

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pkl` | str | 自动最新 | full_pred.pkl 路径 |
| `--output` | str | pkl 父目录 | 输出目录 |

生成 6×4 网格的 24 城新增病例时序图（实际值 vs 预测值），含封城竖线标记。控制台输出各城市 R² 排名。

**示例：**
```bash
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_city_timeseries.py
PYTHONIOENCODING=utf-8 ...python.exe experiments/plot_city_timeseries.py --pkl <path_to_full_pred.pkl>
```

---

## 跨脚本共享参数

| 参数 | 值 | 定义位置 |
|------|-----|----------|
| lookback | 7 天 | 各脚本 `main()` |
| predict | 3 天 | 各脚本 `main()` |
| train/val/test | 60%/20%/20% | 各脚本 `main()` |
| batch_size | min(8, train_size) | 各脚本 `main()` |
| hidden_dim | 16 | 各脚本 `main()` |
| 数据目录 | `E:\Claude code\KAN+\data\合并数据` | `exp_lib/config.py` |
| 结果目录 | `E:\Claude code\KAN+\result` | `exp_lib/config.py` |
| 归一化器 | CityLevelLogMinMaxScaler | `exp_lib/data.py` |
| 默认种子 | 5780 | `exp_lib/config.py` |

## 输出目录结构

```
result/ablation_study_v9_{timestamp}/
├── models/               # 模型权重 .pth
├── results/              # _result.pkl + summary.json
├── figures/              # 单模型图表
└── visualizations/       # 聚合可视化
    ├── architecture/     # 模型结构图
    ├── training/         # 训练曲线
    ├── prediction/       # 预测散点/时序
    ├── parameters/       # β/Rt/Contact
    ├── attention/        # 注意力热力图
    ├── residuals/        # 残差/Q-Q
    ├── error/            # 误差自相关
    ├── evolution/        # 参数演化热力图
    ├── correlation/      # 参数-特征相关
    └── comparison/       # 消融对比/封城参数
```
