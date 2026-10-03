# 工作日志

- 2026-05-24: 克隆项目，适配环境，首次运行 7 模型消融
- 2026-05-25~27: 数据延阔至 2019-11-01，预处理脚本
- 2026-06-09: 通读论文；补齐数据；3种子消融完成 (M_Graphormer 最优 0.7547)
- 2026-06-09: **代码拆分** — exp_chronological.py (5058行) → exp_lib/ (9模块) + 4入口脚本
- 2026-06-16: **KAN vs MLP 对照实验** — 新增 run_kan_vs_mlp.py、plot_city_timeseries.py；修复 M_Graphormer_Baseline/NoPhysics_KAN_Graphormer 子类 **kwargs 传参 bug；Full_KAN_Graphormer R²=0.7414 vs M_Graphormer_Baseline 0.7285 (+0.0129)，KAN 泛化更好、β 更接近文献值 0.46
