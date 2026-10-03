"""
合并 2019-11 ~ 2020-03 数据为 exp_chronological.py 可直接读取的格式
输出到 data/合并数据/ — 与原版文件名和格式完全一致
"""
import pandas as pd
import numpy as np
import os
import shutil

BASE = r'E:\Claude code\KAN+\data'
SRC = os.path.join(BASE, '原始数据')
NEW = os.path.join(BASE, '新增数据')
OUT = os.path.join(BASE, '合并数据')
os.makedirs(OUT, exist_ok=True)

TARGET_CITIES = ['上海市', '北京市', '南京市', '厦门市', '哈尔滨市', '大连市', '天津市',
                 '宁波市', '广州市', '成都市', '无锡市', '杭州市', '武汉市', '沈阳市',
                 '济南市', '深圳市', '石家庄市', '苏州市', '西安市', '郑州市', '重庆市',
                 '长春市', '长沙市', '青岛市']

COLS_FLU = ['日期 (Date)', '城市代码 (City_Code)', '城市名称 (City_Name)',
            '所属省份 (Province)', '省_该月总病例 (Province_Monthly_Case)',
            '城市_人口占比 (City_Pop_Ratio)', '城市_日估算病例 (City_Daily_Case)',
            '发病率(1/10万)', '省人口公式计算（人）(与常住人口一致)',
            '区域标识 (Region)(南方=1，北方=0)', 'City_Monthly_Case']

# ============ 1. Flu data: merge + reorder columns ============
print('=== Merging flu data ===')
df_old_flu = pd.read_excel(os.path.join(SRC, '01_Influenza_Target_Filled.xlsx'))
df_old_flu['日期 (Date)'] = pd.to_datetime(df_old_flu['日期 (Date)'])

df_new_flu = pd.read_excel(os.path.join(NEW, '01_Influenza_Target_extended.xlsx'))
df_new_flu['日期 (Date)'] = pd.to_datetime(df_new_flu['日期 (Date)'])

# Use extended PCHIP for full period (consistent positivity curve avoids boundary jump)
df_merged_flu = df_new_flu.copy()

# Fill any missing static columns from old data
for col in ['城市代码 (City_Code)', '所属省份 (Province)', '省人口公式计算（人）(与常住人口一致)']:
    if df_merged_flu[col].isna().any():
        mapping = df_old_flu[['城市名称 (City_Name)', col]].drop_duplicates().set_index('城市名称 (City_Name)')
        df_merged_flu[col] = df_merged_flu.apply(
            lambda r: mapping.loc[r['城市名称 (City_Name)'], col]
            if pd.isna(r[col]) and r['城市名称 (City_Name)'] in mapping.index else r[col],
            axis=1
        )

dates = sorted(df_merged_flu['日期 (Date)'].unique())
print(f'  Dates: {dates[0].strftime("%Y-%m-%d")} ~ {dates[-1].strftime("%Y-%m-%d")}, {len(dates)} days')
print(f'  Cities: {df_merged_flu["城市名称 (City_Name)"].nunique()}')
print(f'  Rows: {len(df_merged_flu)}')

# Drop helper if exists
df_merged_flu = df_merged_flu.drop(columns=['YearMonth'], errors='ignore')

# Reorder columns to match original
df_merged_flu = df_merged_flu[COLS_FLU]
out_flu = os.path.join(OUT, '01_Influenza_Target_Filled.xlsx')
df_merged_flu.to_excel(out_flu, index=False)
print(f'  -> {out_flu}')

# Spot check: Beijing cases across the merge point
bj = df_merged_flu[df_merged_flu['城市名称 (City_Name)'] == '北京市']
bj_jan = bj[(bj['日期 (Date)'] >= '2020-01-08') & (bj['日期 (Date)'] <= '2020-01-12')]
print('  Beijing around merge:')
for _, r in bj_jan.iterrows():
    print(f'    {r["日期 (Date)"].strftime("%Y-%m-%d")}: {r["城市_日估算病例 (City_Daily_Case)"]:.1f}')

# ============ 2. Final dataset: merge ============
print('\n=== Merging Final_Thesis_Dataset ===')
df_old_env = pd.read_excel(os.path.join(SRC, 'Final_Thesis_Dataset_2020.xlsx'))
df_old_env['日期'] = pd.to_datetime(df_old_env['日期'])

df_new_env = pd.read_excel(os.path.join(NEW, 'Final_Thesis_Dataset_extended.xlsx'))
df_new_env['日期'] = pd.to_datetime(df_new_env['日期'])

# Merge old + new: take 2019-11 to 2020-01-09 from new, 2020-01-10 onward from old
# (Old env data has better bus/metro coverage for 2020 period)
cutoff = pd.Timestamp('2020-01-10')
df_merged_env = pd.concat([
    df_new_env[df_new_env['日期'] < cutoff],
    df_old_env
], ignore_index=True)

# Sort
df_merged_env = df_merged_env.sort_values(['日期', '城市']).reset_index(drop=True)
out_env = os.path.join(OUT, 'Final_Thesis_Dataset_2020.xlsx')
df_merged_env.to_excel(out_env, index=False)
print(f'  -> {out_env}')
print(f'  Rows: {len(df_merged_env)}')
print(f'  Dates: {df_merged_env["日期"].min().strftime("%Y-%m-%d")} ~ {df_merged_env["日期"].max().strftime("%Y-%m-%d")}')

# ============ 3. City features: copy ============
print('\n=== Copying city features ===')
shutil.copy2(os.path.join(SRC, '02_City_Features.xlsx'),
             os.path.join(OUT, '02_City_Features.xlsx'))
print(f'  -> {os.path.join(OUT, "02_City_Features.xlsx")}')

# ============ 4. Migration: use flow_tensor for 2019 + original for 2020 ============
print('\n=== Extending migration matrices ===')
xls = pd.ExcelFile(os.path.join(SRC, '03_Migration_Matrices_Outflow_Normalized.xlsx'))

# Load original 2020 sheets
sheets_data = {}
for sheet in xls.sheet_names:
    sheets_data[sheet] = pd.read_excel(xls, sheet_name=sheet, index_col=0)

# Load flow_tensor for 2019-11~12 (24×24 O-D, absolute flows)
flow_path = os.path.join(NEW, 'flow_tensor_2019_v3(1).xlsx')
if os.path.exists(flow_path):
    df_flow = pd.read_excel(flow_path)
    df_flow['日期'] = pd.to_datetime(df_flow['日期'])
    flow_dates = sorted(df_flow['日期'].unique())
    print(f'  Using flow_tensor: {len(flow_dates)} days, {flow_dates[0].strftime("%Y-%m-%d")} ~ {flow_dates[-1].strftime("%Y-%m-%d")}')

    for dt in flow_dates:
        sheet_name = f'D{dt.strftime("%Y%m%d")}'
        day_flow = df_flow[df_flow['日期'] == dt]
        # Pivot to 24×24 matrix
        mat = day_flow.pivot_table(index='起点城市', columns='终点城市', values='流量', fill_value=0.0)
        # Ensure all target cities present
        mat = mat.reindex(index=TARGET_CITIES, columns=TARGET_CITIES, fill_value=0.0)
        # Row-normalize
        row_sums = mat.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        mat = mat.div(row_sums, axis=0)
        sheets_data[sheet_name] = mat

    # Fill 2020-01-01~09 gap: use 2019-12-31
    gap_start = pd.Timestamp('2020-01-01')
    gap_end = pd.Timestamp('2020-01-09')
    proxy_mat = sheets_data.get('D20191231', sheets_data['D20200110'])
    for dt in pd.date_range(gap_start, gap_end):
        sname = f'D{dt.strftime("%Y%m%d")}'
        if sname not in sheets_data:
            sheets_data[sname] = proxy_mat.copy()
    print(f'  Filled 2020-01-01~09 gap with D20191231 proxy')
else:
    print(f'  WARNING: flow_tensor not found at {flow_path}, using 2020-01-10 proxy')
    proxy_mat = sheets_data['D20200110'].copy()
    for dt in pd.date_range('2019-11-01', '2020-01-09'):
        sheets_data[f'D{dt.strftime("%Y%m%d")}'] = proxy_mat.copy()

# Write extended file
out_mig = os.path.join(OUT, '03_Migration_Matrices_Outflow_Normalized.xlsx')
with pd.ExcelWriter(out_mig, engine='openpyxl') as writer:
    for name in sorted(sheets_data.keys()):
        sheets_data[name].to_excel(writer, sheet_name=name)
print(f'  -> {out_mig}')
sheets_2019 = [s for s in sheets_data if '2019' in s]
sheets_2020 = [s for s in sheets_data if '2020' in s]
print(f'  Sheets: {len(sheets_data)} total (D2019: {len(sheets_2019)}, D2020: {len(sheets_2020)})')
if os.path.exists(flow_path):
    print('  Note: D2019 sheets from flow_tensor, D2020 from original Baidu data')

# Also need to update exp_chronological.py sheet filter.
# Current code: sheet_names = [name for name in xls.sheet_names if name.startswith('D2020')]
# Need to change to include D2019, or rename sheets to D2020*
# For now, rename the 2019 sheets to a format the code can read:
# D2020MMDD where MMDD maps to 2019 dates. No — this is too hacky.
# Better: note the required code change.

print('\n=== Done! ===')
print(f'All files in: {OUT}')
print('To use, change exp_chronological.py:')
print('  Line 54:  BASE_DIR = r"E:\\Claude code\\KAN+\\data\\合并数据"')
print('  Line ~4416: sheet filter → n.startswith("D")  (allow D2019+D2020 sheets)')
