"""
为 4 种 OD 估计方法 (gravity/ipf/mean/snapshot) 各生成一套合并数据
只替换 2019-11~12 的迁徙矩阵，2020 年沿用原 Baidu 数据
"""
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

SRC_OD_DIR = r'E:\PyCharm 2026.1.1\script\paper_figures'
ORIG_MERGED = r'E:\Claude code\KAN+\data\合并数据'
OUT_BASE = r'E:\Claude code\KAN+\data'

TARGET_CITIES = ['上海市', '北京市', '南京市', '厦门市', '哈尔滨市', '大连市', '天津市',
                 '宁波市', '广州市', '成都市', '无锡市', '杭州市', '武汉市', '沈阳市',
                 '济南市', '深圳市', '石家庄市', '苏州市', '西安市', '郑州市', '重庆市',
                 '长春市', '长沙市', '青岛市']

OD_FILES = {
    'gravity': 'gravity_2019_OD.xlsx',
    'ipf': 'ipf_2019_OD.xlsx',
    'mean': 'mean_2019_OD.xlsx',
    'snapshot': 'snapshot_2019_OD.xlsx',
}


def generate_date_mapping():
    """day 1-61 → 2019-11-01 ~ 2019-12-31"""
    mapping = {}
    start = datetime(2019, 11, 1)
    for day_idx in range(1, 62):
        date = start + timedelta(days=day_idx - 1)
        sheet_name = f'D{date.strftime("%Y%m%d")}'
        mapping[day_idx] = sheet_name
    return mapping


def build_migration_from_od(od_path, date_mapping):
    """从长格式 OD 文件生成 {sheet_name: 24x24 DataFrame} 字典"""
    df = pd.read_excel(od_path)
    sheets = {}

    for day_idx, sheet_name in date_mapping.items():
        day_data = df[df['day'] == day_idx]
        mat = day_data.pivot_table(
            index='origin', columns='destination', values='flow', fill_value=0.0
        )
        # 确保行列都包含所有目标城市
        mat = mat.reindex(index=TARGET_CITIES, columns=TARGET_CITIES, fill_value=0.0)
        # 行归一化
        row_sums = mat.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        mat = mat.div(row_sums, axis=0)
        sheets[sheet_name] = mat

    return sheets


def build_merged_data(od_name, od_sheets):
    """为一种 OD 方法创建完整合并数据目录"""
    out_dir = os.path.join(OUT_BASE, f'合并数据_{od_name}')
    os.makedirs(out_dir, exist_ok=True)

    # 1. 复制不变的文件
    for fname in ['01_Influenza_Target_Filled.xlsx', '02_City_Features.xlsx',
                   'Final_Thesis_Dataset_2020.xlsx', 'statement.md']:
        src = os.path.join(ORIG_MERGED, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # 2. 读取原始 2020 迁徙数据
    orig_mig = pd.ExcelFile(os.path.join(ORIG_MERGED, '03_Migration_Matrices_Outflow_Normalized.xlsx'))
    orig_sheets = {}
    for sheet in orig_mig.sheet_names:
        if sheet.startswith('D2020'):
            orig_sheets[sheet] = pd.read_excel(orig_mig, sheet_name=sheet, index_col=0)

    # 3. 合并 2019 (OD) + 2020 (Baidu)
    all_sheets = {}
    all_sheets.update(od_sheets)  # D2019 sheets
    all_sheets.update(orig_sheets)  # D2020 sheets

    # 4. 写入合并后的迁徙文件
    out_mig = os.path.join(out_dir, '03_Migration_Matrices_Outflow_Normalized.xlsx')
    with pd.ExcelWriter(out_mig, engine='openpyxl') as writer:
        for name in sorted(all_sheets.keys()):
            all_sheets[name].to_excel(writer, sheet_name=name)

    d2019 = [s for s in all_sheets if '2019' in s]
    d2020 = [s for s in all_sheets if '2020' in s]
    print(f'  [{od_name}] {out_dir}')
    print(f'    Sheets: {len(all_sheets)} total (D2019: {len(d2019)}, D2020: {len(d2020)})')
    return out_dir


def main():
    print('=' * 60)
    print('Building 4 OD variant merged datasets')
    print('=' * 60)

    date_mapping = generate_date_mapping()
    print(f'\nDate mapping: day 1→{date_mapping[1]}, day 61→{date_mapping[61]}')

    for od_name, filename in OD_FILES.items():
        od_path = os.path.join(SRC_OD_DIR, filename)
        print(f'\n--- Processing {od_name} ({filename}) ---')

        # Build 2019 sheets from OD file
        od_sheets = build_migration_from_od(od_path, date_mapping)
        print(f'  Generated {len(od_sheets)} D2019 sheets from OD file')

        # Quick sanity check
        first_sheet = list(od_sheets.keys())[0]
        mat = od_sheets[first_sheet]
        row_sums = mat.sum(axis=1)
        print(f'  {first_sheet}: shape={mat.shape}, row_sum_range=[{row_sums.min():.6f}, {row_sums.max():.6f}]')

        # Create merged data directory
        build_merged_data(od_name, od_sheets)

    print(f'\n{"=" * 60}')
    print('Done! Created 4 data directories:')
    for od_name in OD_FILES:
        print(f'  data/合并数据_{od_name}/')
    print(f'\nOriginal remains at: data/合并数据/')


if __name__ == '__main__':
    main()
