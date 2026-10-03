# 合并数据目录说明

> 路径：`data/合并数据/`
> 时间范围：**2019-11-01 ~ 2020-03-15**（136 天）
> 城市：24 个目标城市
> 生成脚本：`code/code/preprocessing/merge_data.py`

---

## 文件清单

```
data/合并数据/
├── 01_Influenza_Target_Filled.xlsx          # 流感日估算病例
├── 02_City_Features.xlsx                    # 城市静态特征
├── 03_Migration_Matrices_Outflow_Normalized.xlsx  # 城间迁徙矩阵
└── Final_Thesis_Dataset_2020.xlsx           # 环境/气象/搜索指数
```

---

## 1. 流感日估算病例

**文件**：`01_Influenza_Target_Filled.xlsx`
**规格**：3264 行 × 11 列，136 天 × 24 城

| 列名 | 说明 |
|------|------|
| `日期 (Date)` | 2019-11-01 ~ 2020-03-15 |
| `城市代码 (City_Code)` | 6 位行政区划代码 |
| `城市名称 (City_Name)` | 24 个目标城市名（含"市"） |
| `所属省份 (Province)` | 省会城市为 NaN |
| `省_该月总病例 (Province_Monthly_Case)` | 省级月度发病数 |
| `城市_人口占比 (City_Pop_Ratio)` | 城市占省人口比例 |
| `城市_日估算病例 (City_Daily_Case)` | **目标变量**：PCHIP 降尺度日病例 |
| `发病率(1/10万)` | 日发病率 |
| `省人口公式计算（人）` | 省常住人口 |
| `区域标识 (Region)` | 南方=1，北方=0 |
| `City_Monthly_Case` | 城市月度总病例 |

**数据来源**：
- 2019-11~12：省级月发病数（2019 年全国分地区统计）→ PCHIP 降尺度 + 流感阳性率加权
- 2020-01~03：原始 01 表（同 `原始数据/`）

**读取示例**：
```python
import pandas as pd
df = pd.read_excel(r'data\合并数据\01_Influenza_Target_Filled.xlsx')
df['日期 (Date)'] = pd.to_datetime(df['日期 (Date)'])
target = df['城市_日估算病例 (City_Daily_Case)']  # 预测目标
```

---

## 2. 城市静态特征

**文件**：`02_City_Features.xlsx`
**规格**：24 行 × 11 列

| 列名 | 说明 |
|------|------|
| `城市名称` | 24 城 |
| `城市代码` | 6 位行政区划代码 |
| `2019市常住人口(万人)` | 户籍+常住 |
| `人口密度(人/km²)` | 计算特征用 |
| `城市等级` | 1=一线, 2=二线, 3=其他 |
| `区域标识 (Region)` | 南方=1，北方=0 |
| `供暖标识` | 有集中供暖=1 |
| `GDP(亿元)` | 2019 年 |
| ... | 其他辅助列 |

**读取示例**：
```python
df = pd.read_excel(r'data\合并数据\02_City_Features.xlsx')
```

---

## 3. 城间迁徙矩阵

**文件**：`03_Migration_Matrices_Outflow_Normalized.xlsx`
**规格**：136 个 sheet，每个 24×24

| 属性 | 值 |
|------|-----|
| Sheet 命名 | `D20191101` ~ `D20200315`（D+YYYYMMDD） |
| D2019 | 61 sheet（2019-11-01 ~ 2019-12-31） |
| D2020 | 75 sheet（2020-01-01 ~ 2020-03-15） |
| 行和 | 1.0（行归一化，每行代表出发城市的迁出概率分布） |
| 对角线 | 0（百度只统计跨城流动） |

**数据来源**：
- D2019：来自 `data/新增数据/flow_tensor_2019_v3(1).xlsx`（重力模型从高德总迁入/迁出指数 + 2020年 O-D 模板重构的 24×24 城间绝对流量 → 行归一化）
- D2020-01-01~09：D20191231 前向填充（proxy）
- D2020-01-10~03-15：百度迁徙原始数据（同 `原始数据/`）

**读取示例**：
```python
import pandas as pd

xls = pd.ExcelFile(r'data\合并数据\03_Migration_Matrices_Outflow_Normalized.xlsx')
# 获取某一天的迁徙矩阵
df_mat = pd.read_excel(xls, sheet_name='D20200110', index_col=0)
# df_mat: 24×24, 行=出发城市, 列=目的城市, 值=迁移概率

# 遍历所有日期
sheets = [s for s in xls.sheet_names if s.startswith('D')]
for s in sheets:
    mat = pd.read_excel(xls, sheet_name=s, index_col=0)
    # mat.values: 归一化矩阵
```

---

## 4. 环境/气象/搜索指数

**文件**：`Final_Thesis_Dataset_2020.xlsx`
**规格**：3264 行 × 15 列，136 天 × 24 城

> 注意：城市名列中存在重复名（如"上海"和"上海市"），2019-11~12 用简称，2020-01~03 用全称。代码中通过 `preprocess_intra_city_data()` 统一处理。

| 列名 | 说明 | 覆盖率 |
|------|------|:---:|
| `城市` | 城市名（简称/全称混用） | 100% |
| `日期` | 2019-11-01 ~ 2020-03-15 | 100% |
| `最终气温(℃)` | NOAA GSOD 日均气温 | 100% |
| `相对湿度(%)` | NOAA GSOD | ~100% |
| `降水量(mm)` | NOAA GSOD | ~90% |
| `平均风速(m/s)` | NOAA GSOD | 100% |
| `海平面气压(hPa)` | NOAA GSOD（仅 15/24 城有测站） | 62% |
| `质量等级` | AQI 等级文字 | 100% |
| `PM2.5` | AQI 细颗粒物 | 100% |
| `PM10` | AQI 可吸入颗粒物 | 100% |
| `气温来源` | `NOAA_GSOD` 或遗留来源 | 100% |
| `天气状况(白天/夜间)` | 文字描述 | 100% |
| `Unnamed: 12` | 全 NaN（占位列） | 0% |
| `地铁百度搜索指数` | 百度指数（城内出行代理变量） | 100% |
| `公交百度搜索指数` | 百度指数（城内出行代理变量） | 100% |

**数据来源**：
- 气象：NOAA GSOD（`data/新增数据/NOAA_Weather_24Cities.xlsx`）
- AQI：天气后报爬取（`data/新增数据/08_AQI_201911_202003.xlsx`）
- 天气状况：天气后报爬取（`data/新增数据/07_Weather_201911_202003.xlsx`）
- 公交/地铁：百度指数（`data/新增数据/公交.xlsx`、`地铁.xlsx`，24 城 × 71 天）
- 2019-11~12 的公交地铁来自拟合/爬取，上海/天津已补齐
- 2020-01-10 以后的环境数据同 `原始数据/`

**读取示例**：
```python
df = pd.read_excel(r'data\合并数据\Final_Thesis_Dataset_2020.xlsx')
df['日期'] = pd.to_datetime(df['日期'])
# 获取北京的气温和PM2.5
bj = df[df['城市'].isin(['北京市', '北京'])]
```

---

## 主程序调用

`exp_chronological.py` 已配置为读取此目录：

```python
# Line 54
BASE_DIR = r"E:\Claude code\KAN+\data\合并数据"
FILE_FLU_TARGET    = os.path.join(BASE_DIR, "01_Influenza_Target_Filled.xlsx")
FILE_CITY_FEAT     = os.path.join(BASE_DIR, "02_City_Features.xlsx")
FILE_MIG_PATH      = os.path.join(BASE_DIR, "03_Migration_Matrices_Outflow_Normalized.xlsx")
FILE_DATASET_2020  = os.path.join(BASE_DIR, "Final_Thesis_Dataset_2020.xlsx")

# Line ~4416 — sheet filter 已改为允许 D2019+D2020
sheet_names = [n for n in xls.sheet_names if n.startswith('D')]
```

---

## 与原始数据的差异

| 维度 | 原始数据 (66天) | 合并数据 (136天) |
|------|:---:|:---:|
| 时间范围 | 2020-01-10 ~ 03-15 | **2019-11-01** ~ 2020-03-15 |
| 迁徙矩阵 | 66 sheet (D2020) | 136 sheet (D2019 + D2020) |
| D2019 迁徙来源 | — | flow_tensor（重力模型重构） |
| 上海/天津公交地铁 | — | 已补齐 |
| 训练样本 | 57 | 127 |
| 数据划分 | 60%/20%/20% | 60%/20%/20%（训练+验证随机打乱） |

---

*生成日期：2026-06-09*
