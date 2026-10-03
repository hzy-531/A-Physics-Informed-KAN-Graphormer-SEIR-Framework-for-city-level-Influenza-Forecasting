"""
全局配置与参数 — 从 exp_chronological.py 提取
所有实验脚本共享的常量、路径、城市列表等
"""
import os
import random
import numpy as np
import torch
import pandas as pd

# ==========================================
# 随机种子 (默认值，运行时可通过 set_seed 覆盖)
# ==========================================
SEED = 5780

def set_seed(seed: int):
    """设置全局随机种子确保可重复性"""
    global SEED
    SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✅ 随机种子已固定为: {seed}")

# 初始化默认种子
set_seed(SEED)

# ==========================================
# 路径配置
# ==========================================
BASE_DIR = r"E:\Claude code\KAN+\data\合并数据"
FILE_FLU_TARGET = os.path.join(BASE_DIR, "01_Influenza_Target_Filled.xlsx")
FILE_CITY_FEAT = os.path.join(BASE_DIR, "02_City_Features.xlsx")
FILE_MIG_PATH = os.path.join(BASE_DIR, "03_Migration_Matrices_Outflow_Normalized.xlsx")
FILE_DATASET_2020 = os.path.join(BASE_DIR, "Final_Thesis_Dataset_2020.xlsx")

def create_output_dir(base_result_dir=None):
    """创建带时间戳的输出目录"""
    if base_result_dir is None:
        base_result_dir = r"E:\Claude code\KAN+\result"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_result_dir, f"ablation_study_v9_{timestamp}")
    for sub in ["models", "figures", "results", "visualizations"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    return output_dir

# 全局输出目录 (由入口脚本设置)
OUTPUT_DIR = None

# ==========================================
# 城市列表与等级
# ==========================================
TARGET_CITIES = [
    '北京市', '天津市', '上海市', '重庆市', '广州市', '深圳市', '西安市', '成都市',
    '武汉市', '杭州市', '南京市', '苏州市', '无锡市', '郑州市', '长沙市', '沈阳市',
    '大连市', '青岛市', '济南市', '宁波市', '厦门市', '哈尔滨市', '长春市', '石家庄市'
]

TIER_1_CITIES = ['北京市', '上海市', '广州市', '深圳市']
TIER_2_CITIES = ['天津市', '重庆市', '成都市', '武汉市', '南京市', '杭州市', '西安市', '郑州市']

# ==========================================
# 封城信息
# ==========================================
LOCKDOWN_INFO = {
    '武汉市': {'start': '2020-01-23', 'end': '2020-04-08', 'strength': 0.8},
    '北京市': {'start': None, 'end': None, 'strength': 0.3},
    '石家庄市': {'start': '2020-01-24', 'end': '2020-02-09', 'strength': 0.6},
    '哈尔滨市': {'start': '2020-02-04', 'end': '2020-03-04', 'strength': 0.5}
}

# ==========================================
# matplotlib 全局设置
# ==========================================
import matplotlib.pyplot as plt
import warnings

plt.style.use(['default', 'seaborn-v0_8-darkgrid'])
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
