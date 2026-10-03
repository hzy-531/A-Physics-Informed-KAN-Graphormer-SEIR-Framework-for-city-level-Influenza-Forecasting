"""
exp_lib — KAN-Graphormer 消融实验共享库
从 exp_chronological.py 拆分而来

模块结构:
- config:   全局配置、路径、城市列表、随机种子
- layers:   RobustKANLinear, GatedTCN, GraphormerLayer, GAT, 空间编码
- physics:  SEIR ODE 微分方程、RK4 求解器、R0 计算
- data:     数据加载、归一化器、时序增强、数据集类
- models:   全部 7 个消融模型
- loss:     CurriculumScientificLoss
- trainer:  ImprovedAblationStudyManager (训练+评估+全时段预测)
- visualizer: ImprovedActualDataVisualizer (全部图表)
"""

from .config import (
    SEED, set_seed, create_output_dir, OUTPUT_DIR,
    BASE_DIR, FILE_FLU_TARGET, FILE_CITY_FEAT, FILE_MIG_PATH, FILE_DATASET_2020,
    TARGET_CITIES, TIER_1_CITIES, TIER_2_CITIES, LOCKDOWN_INFO,
)

from .layers import (
    RobustKANLinear, StandardMLP, GatedTCN,
    RBFSpatialEncoding, GraphormerLayer, ImprovedGATLayer,
    compute_diffusion_distance,
)

from .physics import (
    compute_derivatives_seir, age_structured_seir_step, calculate_R0_NGM,
)

from .data import (
    DataCollector,
    CityLevelLogMinMaxScaler, CityLevelLinearMinMaxScaler,
    TemporalFeatureEnhancer, ImprovedTemporalDataset,
    preprocess_intra_city_data, get_smooth_age_ratios,
    compute_lockdown_factor, add_city_specific_features,
    load_raw_data_only, build_enhanced_dataset_with_scalers,
)

from .models import (
    EnhancedFull_Graphormer_V3,
    EnhancedMLP_Baseline_V3,
    EnhancedLSTM_Baseline_V3,
    EnhancedGAT_Baseline_V3,
    EnhancedKAN_Only_V3,
    M_Graphormer_Baseline,
    NoPhysics_KAN_Graphormer,
    MODEL_DEFINITIONS,
)

from .loss import CurriculumScientificLoss

from .trainer import ImprovedAblationStudyManager

from .visualizer import ImprovedActualDataVisualizer
