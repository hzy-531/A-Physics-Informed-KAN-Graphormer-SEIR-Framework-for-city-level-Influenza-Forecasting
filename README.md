# KAN-Graphormer-SEIR

**A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting**

Zhuoyun Hu, Zeyu Zhen, Wen Zhu, Congping Lin
— Huazhong University of Science and Technology, Wuhan, China

This repository accompanies the paper *"A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting."* It contains the source code, training/plotting scripts, and the LaTeX source of the manuscript.

The framework jointly forecasts daily influenza incidence across 24 Chinese cities by combining:

- a compartmental **SEIR** metapopulation transmission model (physics-informed constraint),
- a **Graphormer** module for inter-city mobility/connectivity representation,
- **Kolmogorov–Arnold Networks (KAN)** as learnable nonlinear decoders.

Beyond point forecasts, it infers time-varying epidemiological parameters ($\beta_i(t)$, $\gamma_i(t)$, $c_i(t)$) and estimates a perturbation-based outbreak sensitivity score via Monte Carlo inference.

---

## Repository structure

```
.
├── paper/                     # LaTeX source + figures used in the manuscript
│   ├── KAN.tex
│   └── *.pdf                  # 6 auto-generated result figures
├── code/
│   ├── exp_lib/               # shared library (config, layers, physics, data,
│   │                          #   models, loss, trainer, visualizer)
│   ├── experiments/           # training + plotting entry points
│   ├── preprocessing/         # data assembly pipeline
│   └── analysis/              # interpretability analysis
├── docs/                      # architecture / experiment design / results notes
├── thesis/                    # thesis statement
├── data/                      # data format description (data files not included)
├── requirements.txt
└── LICENSE
```

## Environment

- Python 3.9+ (developed with a PyTorch `venv`)
- PyTorch (CUDA or CPU), NumPy, Pandas, Matplotlib, SciPy, scikit-learn, NetworkX, Seaborn, tqdm, openpyxl, requests

```bash
pip install -r requirements.txt
```

## Data

The training data are four `.xlsx` files under `data/合并数据/` (see `data/statement.md` for the schema):

- `01_Influenza_Target_Filled.xlsx` — daily influenza incidence targets
- `02_City_Features.xlsx` — static city-level covariates
- `03_Migration_Matrices_Outflow_Normalized.xlsx` — inter-city mobility matrices
- `Final_Thesis_Dataset_2020.xlsx` — assembled features

The raw data files are **not** included in this repository. Data paths are hardcoded in `code/exp_lib/config.py` (`BASE_DIR`, `base_result_dir`); update them to point to your local copies before running.

## Reproduction

All commands run from the `code/` directory (the scripts insert their parent directory into `sys.path` to import `exp_lib`). On Windows, prefix commands with `PYTHONIOENCODING=utf-8` to avoid GBK console errors.

```bash
cd code

# Quick sanity check (1 model × 1 seed)
python experiments/run_single.py --epochs 20

# KAN vs MLP decoder comparison (2 models × 1 seed)
python experiments/run_kan_vs_mlp.py --epochs 30

# Full ablation (7 models × 1 seed)
python experiments/run_ablation.py

# Main experiments (ablation × 3 seeds)
python experiments/run_3seeds.py

# 20-seed runs used for the paper's table
python experiments/run_20seeds_full_final.py
python experiments/run_20seeds_6models.py
python experiments/run_20seeds_align_capacity.py

# Regenerate the paper's result figures (read existing .pkl, seconds)
python experiments/regen_scatter.py
python experiments/regen_city_level.py
python experiments/regen_evolution.py
python experiments/regen_lockdown.py
python experiments/regen_outbreak.py
python experiments/regen_timeseries.py
python experiments/plot_epi_stability.py
python experiments/plot_outbreak_merged_panels.py

# Aggregate metrics across seeds into mean ± SE
python experiments/multi_seed_analysis.py
python experiments/convert_std_to_sem.py
```

Figure outputs are written under `result/` (ignored by git).

## Figures note

The manuscript references two **architecture diagrams** that were not found among the project files and are therefore **not** included here:

- `KAN-Graphormer混合预测模型总体架构与信息流转图.pdf` (overall architecture & information flow)
- `Graphormer层的内部结构图.pdf` (Graphormer layer internals)

These must be added to `paper/` (or generated) before compiling `KAN.tex`.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite:

> Hu Z., Zhen Z., Zhu W., Lin C. *A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting.*
