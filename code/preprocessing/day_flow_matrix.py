import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d, PchipInterpolator
from datetime import datetime, timedelta

# ============================================
# 第一部分：构建迁徙动态矩阵 - 验证归一化一致性
# ============================================

# 配置参数
base_path = r"E:\Claude code\KAN+\data\原始数据\百度迁徙数据"
target_cities = ["北京市", "天津市", "上海市", "重庆市", "广州市", "深圳市", "西安市", "成都市", "武汉市", "杭州市",
                 "南京市", "苏州市", "无锡市", "郑州市", "长沙市", "沈阳市", "大连市", "青岛市", "济南市", "宁波市",
                 "厦门市", "哈尔滨市", "长春市", "石家庄市"]

start_date = "2020-01-10"
end_date = "2020-03-15"
date_range = pd.date_range(start=start_date, end=end_date)

# 初始化字典来存储不同视角的矩阵
daily_matrices_outflow = {}  # 迁出视角
daily_matrices_inflow = {}  # 迁入视角
daily_matrices_hybrid = {}  # 混合视角

print(">>> [Part 1] 开始处理迁徙数据，构建动态矩阵...")
for single_date in date_range:
    date_str = single_date.strftime("%Y%m%d")
    outflow_flows = []
    inflow_flows = []

    # 初始化城市总迁出和总迁入
    city_total_outflow = {city: 0.0 for city in target_cities}
    city_total_inflow = {city: 0.0 for city in target_cities}

    for city in target_cities:
        file_name = f"{city}_{date_str}.csv"
        file_path = os.path.join(base_path, file_name)

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # 1. 处理迁出
            df_out = df[(df['rank_type'] == 'move_out_city') & (df['target_city_name'].isin(target_cities))]
            for _, row in df_out.iterrows():
                outflow_flows.append({
                    'date': single_date,
                    'source_city': city,
                    'target_city': row['target_city_name'],
                    'migration_index': row['value']
                })
                city_total_outflow[city] += row['value']

            # 2. 处理迁入
            df_in = df[(df['rank_type'] == 'move_in_city') & (df['target_city_name'].isin(target_cities))]
            for _, row in df_in.iterrows():
                inflow_flows.append({
                    'date': single_date,
                    'source_city': row['target_city_name'],
                    'target_city': city,
                    'migration_index': row['value']
                })
                city_total_inflow[city] += row['value']

        except Exception as e:
            print(f"读取或处理文件 {file_name} 时出错: {e}")
            continue

    if outflow_flows and inflow_flows:
        # --- 迁出矩阵 (行归一化) ---
        df_outflow = pd.DataFrame(outflow_flows)
        matrix_outflow_raw = df_outflow.pivot_table(
            index='source_city', columns='target_city', values='migration_index', fill_value=0.0
        ).reindex(index=target_cities, columns=target_cities, fill_value=0.0)

        matrix_outflow_normalized = matrix_outflow_raw.copy()
        for source_city in target_cities:
            row_sum = matrix_outflow_normalized.loc[source_city].sum()
            if row_sum > 0:
                matrix_outflow_normalized.loc[source_city] /= row_sum

        # --- 迁入矩阵 (列归一化) ---
        df_inflow = pd.DataFrame(inflow_flows)
        matrix_inflow_raw = df_inflow.pivot_table(
            index='source_city', columns='target_city', values='migration_index', fill_value=0.0
        ).reindex(index=target_cities, columns=target_cities, fill_value=0.0)

        matrix_inflow_normalized = matrix_inflow_raw.copy()
        for target_city in target_cities:
            col_sum = matrix_inflow_normalized[target_city].sum()
            if col_sum > 0:
                matrix_inflow_normalized[target_city] /= col_sum

        # --- 混合矩阵 (几何平均) ---
        matrix_hybrid = pd.DataFrame(0.0, index=target_cities, columns=target_cities)
        for source_city in target_cities:
            for target_city in target_cities:
                outflow_val = matrix_outflow_raw.loc[source_city, target_city]
                inflow_val = matrix_inflow_raw.loc[source_city, target_city]
                if outflow_val > 0 and inflow_val > 0:
                    matrix_hybrid.loc[source_city, target_city] = np.sqrt(outflow_val * inflow_val)
                else:
                    matrix_hybrid.loc[source_city, target_city] = max(outflow_val, inflow_val)

        matrix_hybrid_normalized = matrix_hybrid.copy()
        for source_city in target_cities:
            row_sum = matrix_hybrid_normalized.loc[source_city].sum()
            if row_sum > 0:
                matrix_hybrid_normalized.loc[source_city] /= row_sum

        daily_matrices_outflow[date_str] = matrix_outflow_normalized
        daily_matrices_inflow[date_str] = matrix_inflow_normalized
        daily_matrices_hybrid[date_str] = matrix_hybrid_normalized

        # 验证 20200110
        if date_str == "20200110":
            print(f"\n验证20200110归一化一致性:")
            test_pair = ("天津市", "北京市")
            val_out = matrix_outflow_normalized.loc[test_pair]
            val_in = matrix_inflow_normalized.loc[test_pair]
            print(f"  {test_pair[0]}->{test_pair[1]}: OutNorm={val_out:.4f}, InNorm={val_in:.4f}")

    else:
        zero_matrix = pd.DataFrame(0.0, index=target_cities, columns=target_cities)
        daily_matrices_outflow[date_str] = zero_matrix
        daily_matrices_inflow[date_str] = zero_matrix
        daily_matrices_hybrid[date_str] = zero_matrix
        print(f"日期 {date_str} 无有效流动数据，已创建全零矩阵。")

# 保存矩阵
RESULT_DIR = r"E:\Claude code\KAN+\result"
output_paths = {
    "outflow": os.path.join(RESULT_DIR, "03_Migration_Matrices_Outflow_Normalized.xlsx"),
    "inflow": os.path.join(RESULT_DIR, "03_Migration_Matrices_Inflow_Normalized.xlsx"),
    "hybrid": os.path.join(RESULT_DIR, "03_Migration_Matrices_Hybrid_Normalized.xlsx")
}

for matrix_type, file_path in output_paths.items():
    matrices = daily_matrices_outflow if matrix_type == "outflow" else \
        daily_matrices_inflow if matrix_type == "inflow" else daily_matrices_hybrid

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        for date_str, matrix in matrices.items():
            matrix.to_excel(writer, sheet_name=f"D{date_str}")
    print(f"  已保存: {file_path}")

print("迁徙数据处理完成。")

# ============================================
# 第二部分：基于周报趋势的流感数据处理 (改进版)
# ============================================

print("\n>>> [Part 2] 开始处理流感数据，使用南北方周报趋势进行修正...")

# 1. 配置文件路径
file_path_01 = r"E:\Claude code\KAN+\data\原始数据\01_Influenza_Target_Filled.xlsx"  # 原始流感月度表
file_path_02 = r"E:\Claude code\KAN+\data\原始数据\02_City_Features.xlsx"  # 城市特征表(含南北方标识)

# 2. 准备周报趋势数据
# 格式: {周结束日期: 阳性率(%)}
weekly_trend_north = {
    "2020-01-12": 42.3,
    "2020-01-19": 36.5,
    "2020-01-26": 32.5,
    "2020-02-02": 24.2,
    "2020-02-09": 11.5,
    "2020-02-16": 5.9,
    "2020-02-23": 4.1,
    "2020-03-01": 3.5,
    "2020-03-08": 1.4,
    "2020-03-15": 1.7,
}

weekly_trend_south = {
    "2020-01-12": 48.1,
    "2020-01-19": 44.5,
    "2020-01-26": 39.2,
    "2020-02-02": 33.2,
    "2020-02-09": 16.3,
    "2020-02-16": 6.5,
    "2020-02-23": 1.7,
    "2020-03-01": 1,
    "2020-03-08": 1,
    "2020-03-15": 0.7,
}


# 3. 生成日度权重函数
def generate_daily_weights(trend_dict, start_date_str, end_date_str):
    dates = pd.to_datetime(list(trend_dict.keys()))
    values = list(trend_dict.values())

    # 使用 PCHIP 插值保证单调性和平滑性
    interpolator = PchipInterpolator(dates.map(datetime.toordinal), values)

    target_range = pd.date_range(start_date_str, end_date_str)
    daily_weights = interpolator(target_range.map(datetime.toordinal))
    daily_weights = np.maximum(daily_weights, 0)  # 确保非负

    return pd.Series(daily_weights, index=target_range)


print("生成南北方日度趋势曲线...")
daily_weights_north_series = generate_daily_weights(weekly_trend_north, start_date, end_date)
daily_weights_south_series = generate_daily_weights(weekly_trend_south, start_date, end_date)

# 4. 读取并合并数据
df_flu = pd.read_excel(file_path_01)
df_feat = pd.read_excel(file_path_02)

# 确保日期列格式正确
df_flu['日期 (Date)'] = pd.to_datetime(df_flu['日期 (Date)'])

# 计算城市月度总病例 (如果还没计算)
if 'City_Monthly_Case' not in df_flu.columns:
    df_flu['City_Monthly_Case'] = df_flu['省_该月总病例 (Province_Monthly_Case)'] * df_flu[
        '城市_人口占比 (City_Pop_Ratio)']

# 建立城市 -> 区域(南北) 映射
# 假设 02 表中有 "城市名称" 和 "区域标识 (Region)(南方=1，北方=0)"
city_region_map = {}
try:
    # 尝试匹配列名，去除可能的空格
    feat_cols = [c.strip() for c in df_feat.columns]
    df_feat.columns = feat_cols

    # 找到包含 'Region' 或 '区域' 的列名
    region_col = [c for c in df_feat.columns if 'Region' in c or '区域' in c][0]
    name_col = [c for c in df_feat.columns if '城市名称' in c][0]

    for _, row in df_feat.iterrows():
        city_region_map[row[name_col]] = row[region_col]

    print("成功加载城市区域映射 (南方/北方)。")
except Exception as e:
    print(f"警告: 无法从 02 表加载区域映射 ({e})，将默认全部使用南方趋势。")

# 5. 执行基于趋势的插值
print("正在执行加权插值计算...")
df_flu['城市_日估算病例 (City_Daily_Case)'] = np.nan

cities = df_flu['城市名称 (City_Name)'].unique()

for city in cities:
    city_df = df_flu[df_flu['城市名称 (City_Name)'] == city].copy()

    # 确定该城市使用哪条曲线
    is_south = True  # 默认
    if city in city_region_map:
        is_south = (city_region_map[city] == 1)

    weights_series = daily_weights_south_series if is_south else daily_weights_north_series

    # 按月处理
    city_df['YearMonth'] = city_df['日期 (Date)'].dt.to_period('M')

    for ym, group in city_df.groupby('YearMonth'):
        # 获取该月需要分配的总病例数 (取该组第一个值即可，因为同月相同)
        monthly_total = group['City_Monthly_Case'].iloc[0]

        # 获取该月对应的日期索引
        group_dates = group['日期 (Date)']

        # 获取这些天的权重
        # reindex可能会引入NaN (如果日期超出范围)，用0填充
        current_weights = weights_series.reindex(group_dates).fillna(0).values

        total_weight = current_weights.sum()

        if total_weight > 0:
            # 核心公式：按权重比例分配月度总数
            distributed_cases = monthly_total * (current_weights / total_weight)
        else:
            # 权重全为0，平均分配
            distributed_cases = np.full(len(group), monthly_total / len(group))

        # 将结果填回原表
        df_flu.loc[group.index, '城市_日估算病例 (City_Daily_Case)'] = distributed_cases

    # 顺便把 Region 标识也填入 01 表，方便后续模型使用
    df_flu.loc[df_flu['城市名称 (City_Name)'] == city, '区域标识 (Region)(南方=1，北方=0)'] = 1 if is_south else 0

# 6. 保存结果
output_flu_path = os.path.join(r"E:\Claude code\KAN+\result", "01_Influenza_Target_Filled.xlsx")  # 输出到result目录
df_flu.to_excel(output_flu_path, index=False)

print(f"处理完成！")
print(f"1. 迁徙矩阵已保存至: {list(output_paths.values())}")
print(f"2. 优化后的流感日度数据已保存至: {output_flu_path}")
print("   (已包含 '区域标识' 列，并使用周报趋势修正了插值结果)")