# KAN-Graphormer-SEIR

**A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting**

Zhuoyun Hu, Zeyu Zhen, Wen Zhu, Congping Lin
— Huazhong University of Science and Technology, Wuhan, China

This repository accompanies the paper *"A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting."* It contains the source code, training and plotting scripts, and the LaTeX source of the manuscript.

The framework jointly forecasts daily influenza incidence across 24 Chinese cities by combining:

- a compartmental **SEIR** metapopulation transmission model (physics-informed constraint),
- a **Graphormer** module for inter-city mobility/connectivity representation,
- **Kolmogorov–Arnold Networks (KAN)** as learnable nonlinear decoders.

Beyond point forecasts, it infers time-varying epidemiological parameters ($\beta_i(t)$, $\gamma_i(t)$, $c_i(t)$) and estimates a perturbation-based outbreak sensitivity score via Monte Carlo inference.

---

## Repository structure

```
.
├── paper/                     # LaTeX source and manuscript figures
│   ├── KAN.tex
│   └── *.pdf
├── code/
│   ├── exp_lib/               # shared library (config, layers, physics, data,
│   │                          #   models, loss, trainer, visualizer)
│   ├── experiments/           # training and plotting entry points
│   ├── preprocessing/         # data assembly pipeline
│   └── analysis/              # interpretability analysis
├── docs/                      # architecture, experiment design, and results
├── thesis/                    # thesis statement
├── data/                      # data format description
├── requirements.txt
└── LICENSE
```

## Environment

PyTorch, NumPy, Pandas, Matplotlib, SciPy, scikit-learn, NetworkX, Seaborn, tqdm, openpyxl, requests.

```bash
pip install -r requirements.txt
```

## Data

The model reads four spreadsheets under `data/合并数据/` (see `data/statement.md` for the schema):

- `01_Influenza_Target_Filled.xlsx` — daily influenza incidence targets
- `02_City_Features.xlsx` — static city-level covariates
- `03_Migration_Matrices_Outflow_Normalized.xlsx` — inter-city mobility matrices
- `Final_Thesis_Dataset_2020.xlsx` — assembled features

## Usage

All commands run from the `code/` directory.

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

# Regenerate the paper's result figures
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

Figure outputs are written under `result/`.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite:

> Hu Z., Zhen Z., Zhu W., Lin C. *A Physics-Informed KAN-Graphormer-SEIR Framework for City-Level Influenza Forecasting.*
