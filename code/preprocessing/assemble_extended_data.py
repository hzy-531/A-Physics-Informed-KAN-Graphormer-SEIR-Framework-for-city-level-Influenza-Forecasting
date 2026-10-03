"""
将 2019-11-01 ~ 2020-03-15 新增数据组装为 exp_chronological.py 可调用的格式
输出: 01_Influenza_Target_Filled_extended.xlsx + Final_Thesis_Dataset_extended.xlsx
"""
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from datetime import datetime, timedelta
import os

BASE = r'E:\Claude code\KAN+\data'
SRC = os.path.join(BASE, '原始数据')
NEW = os.path.join(BASE, '新增数据')
OUT = NEW  # 输出到新增数据目录

TARGET_CITIES = ['上海市', '北京市', '南京市', '厦门市', '哈尔滨市', '大连市', '天津市',
                 '宁波市', '广州市', '成都市', '无锡市', '杭州市', '武汉市', '沈阳市',
                 '济南市', '深圳市', '石家庄市', '苏州市', '西安市', '郑州市', '重庆市',
                 '长春市', '长沙市', '青岛市']

START_DATE = '2019-11-01'
END_DATE = '2020-03-15'

# ============ 省份 → 城市映射 ============
PROVINCE_CITIES = {
    '北京市': ['北京市'], '天津市': ['天津市'], '上海市': ['上海市'], '重庆市': ['重庆市'],
    '广东省': ['广州市', '深圳市'], '陕西省': ['西安市'], '四川省': ['成都市'],
    '湖北省': ['武汉市'], '浙江省': ['杭州市', '宁波市'],
    '江苏省': ['南京市', '苏州市', '无锡市'], '河南省': ['郑州市'], '湖南省': ['长沙市'],
    '辽宁省': ['沈阳市', '大连市'], '山东省': ['青岛市', '济南市'], '福建省': ['厦门市'],
    '黑龙江省': ['哈尔滨市'], '吉林省': ['长春市'], '河北省': ['石家庄市'],
}

# ============ Part 1: Flu daily estimation ============
def build_extended_flu():
    """构建 2019-11-01 ~ 2020-03-15 流感日估算数据"""
    print('=== Part 1: Building extended flu data ===')

    # 1a. Load existing flu data for static info
    df_old = pd.read_excel(os.path.join(SRC, '01_Influenza_Target_Filled.xlsx'))
    df_old['日期 (Date)'] = pd.to_datetime(df_old['日期 (Date)'])

    # 1b. Load 2019 monthly data
    df_2019 = pd.read_excel(os.path.join(NEW, '2019年全国分地区流行性感冒分月统计数据.xls'),
                            skiprows=1)
    # Rename: [0]=province, [1]=month, [2]=cases, [3]=rate, [4]=deaths, [5]=death_rate
    df_2019.columns = ['省份', '月份', '发病数', '发病率', '死亡数', '死亡率']

    # 1c. Get city-static mappings from old data
    city_info = df_old[['城市代码 (City_Code)', '城市名称 (City_Name)',
                         '所属省份 (Province)', '城市_人口占比 (City_Pop_Ratio)',
                         '区域标识 (Region)(南方=1，北方=0)',
                         '省人口公式计算（人）(与常住人口一致)']].drop_duplicates()

    # For municipalities, 所属省份 is NaN → fill with city name
    city_info['Province_filled'] = city_info['所属省份 (Province)'].fillna(
        city_info['城市名称 (City_Name)'])

    # 1d. Assemble monthly provincial case totals
    target_months = ['2019年11月', '2019年12月', '2020年1月', '2020年2月', '2020年3月']

    # From old data, get monthly totals per city (already computed)
    old_monthly = df_old.groupby(['城市名称 (City_Name)', df_old['日期 (Date)'].dt.month])[
        'City_Monthly_Case'].first().reset_index()
    old_monthly.columns = ['城市', '月份数字', '月总病例']

    # Build new rows for each city × day
    date_range = pd.date_range(START_DATE, END_DATE)
    rows = []

    for city in TARGET_CITIES:
        info = city_info[city_info['城市名称 (City_Name)'] == city]
        if len(info) == 0:
            continue
        info = info.iloc[0]
        province = info['Province_filled']
        city_code = info['城市代码 (City_Code)']
        pop_ratio = info['城市_人口占比 (City_Pop_Ratio)']
        region = info['区域标识 (Region)(南方=1，北方=0)']
        prov_pop = info['省人口公式计算（人）(与常住人口一致)']

        for dt in date_range:
            month_num = dt.month
            year_month = f'{dt.year}年{dt.month}月'

            # Get monthly provincial total
            if month_num >= 1 and dt.year == 2020:
                # From existing data (reliable)
                old_row = old_monthly[(old_monthly['城市'] == city) &
                                       (old_monthly['月份数字'] == month_num)]
                if len(old_row) > 0:
                    prov_total = old_row['月总病例'].iloc[0] / pop_ratio if pop_ratio > 0 else 0
                else:
                    prov_total = 0
            else:
                # From 2019 data
                m19 = df_2019[(df_2019['省份'].str.strip() == province) &
                               (df_2019['月份'] == year_month)]
                if len(m19) > 0:
                    prov_total = m19['发病数'].iloc[0]
                else:
                    prov_total = 0

            city_monthly = prov_total * pop_ratio if pop_ratio > 0 else 0

            rows.append({
                '日期 (Date)': dt,
                '城市代码 (City_Code)': city_code,
                '城市名称 (City_Name)': city,
                '所属省份 (Province)': info['所属省份 (Province)'],
                '省_该月总病例 (Province_Monthly_Case)': prov_total,
                '城市_人口占比 (City_Pop_Ratio)': pop_ratio,
                '区域标识 (Region)(南方=1，北方=0)': region,
                '省人口公式计算（人）(与常住人口一致)': prov_pop,
                'City_Monthly_Case': city_monthly,
            })

    df_flu = pd.DataFrame(rows)
    print(f'  Base rows: {len(df_flu)} ({len(df_flu)//len(TARGET_CITIES)} days × '
          f'{len(df_flu)//len(date_range)} cities)')

    # 1e. PCHIP daily estimation with positivity rates
    weekly_trend_south = {
        "2019-11-03": 5.7, "2019-11-10": 6.9, "2019-11-17": 8.5, "2019-11-24": 11.5,
        "2019-12-01": 16.1, "2019-12-08": 22.7, "2019-12-15": 31.7, "2019-12-22": 40.2,
        "2019-12-29": 46.3, "2020-01-05": 48.4, "2020-01-12": 48.1, "2020-01-19": 44.5,
        "2020-01-26": 39.2, "2020-02-02": 33.2, "2020-02-09": 16.3, "2020-02-16": 6.5,
        "2020-02-23": 1.7, "2020-03-01": 1.0, "2020-03-08": 1.0, "2020-03-15": 0.7,
    }
    weekly_trend_north = {
        "2019-11-03": 1.0, "2019-11-10": 1.6, "2019-11-17": 2.3, "2019-11-24": 4.4,
        "2019-12-01": 9.2, "2019-12-08": 16.1, "2019-12-15": 27.6, "2019-12-22": 38.5,
        "2019-12-29": 43.6, "2020-01-05": 47.0, "2020-01-12": 42.3, "2020-01-19": 36.5,
        "2020-01-26": 32.5, "2020-02-02": 24.2, "2020-02-09": 11.5, "2020-02-16": 5.9,
        "2020-02-23": 4.1, "2020-03-01": 3.5, "2020-03-08": 1.4, "2020-03-15": 1.7,
    }

    def generate_daily_weights(trend_dict):
        dates = pd.to_datetime(list(trend_dict.keys()))
        values = list(trend_dict.values())
        interpolator = PchipInterpolator(dates.map(datetime.toordinal), values)
        target_range = pd.date_range(START_DATE, END_DATE)
        daily = interpolator(target_range.map(datetime.toordinal))
        return pd.Series(np.maximum(daily, 0), index=target_range)

    dw_south = generate_daily_weights(weekly_trend_south)
    dw_north = generate_daily_weights(weekly_trend_north)

    # Apply daily estimation per city
    df_flu['城市_日估算病例 (City_Daily_Case)'] = np.nan
    df_flu['发病率(1/10万)'] = np.nan

    for city in TARGET_CITIES:
        city_mask = df_flu['城市名称 (City_Name)'] == city
        city_df = df_flu[city_mask].copy()
        region_val = city_df['区域标识 (Region)(南方=1，北方=0)'].iloc[0]
        is_south = region_val == 1

        weights = dw_south if is_south else dw_north

        # Process by month
        city_df['YearMonth'] = city_df['日期 (Date)'].dt.to_period('M')

        for ym, group in city_df.groupby('YearMonth'):
            monthly_total = group['City_Monthly_Case'].iloc[0]
            group_dates = group['日期 (Date)']
            current_weights = weights.reindex(group_dates).fillna(0).values
            total_weight = current_weights.sum()

            if total_weight > 0:
                distributed = monthly_total * (current_weights / total_weight)
            else:
                distributed = np.full(len(group), monthly_total / len(group))

            df_flu.loc[group.index, '城市_日估算病例 (City_Daily_Case)'] = distributed

    # Compute incidence rate
    for city in TARGET_CITIES:
        city_mask = df_flu['城市名称 (City_Name)'] == city
        prov_pop = df_flu.loc[city_mask, '省人口公式计算（人）(与常住人口一致)'].iloc[0]
        if prov_pop and prov_pop > 0:
            df_flu.loc[city_mask, '发病率(1/10万)'] = (
                df_flu.loc[city_mask, '城市_日估算病例 (City_Daily_Case)'] / prov_pop * 100000)

    # Save
    out_path = os.path.join(OUT, '01_Influenza_Target_extended.xlsx')
    df_flu.to_excel(out_path, index=False)
    print(f'  Saved: {out_path}')
    print(f'  Date range: {df_flu["日期 (Date)"].min()} ~ {df_flu["日期 (Date)"].max()}')
    print(f'  Total rows: {len(df_flu)}')

    # Quick stats
    for city in ['北京市', '上海市', '广州市']:
        city_data = df_flu[df_flu['城市名称 (City_Name)'] == city]
        total = city_data['城市_日估算病例 (City_Daily_Case)'].sum()
        print(f'  {city}: total={total:.0f}, '
              f'daily avg={city_data["城市_日估算病例 (City_Daily_Case)"].mean():.1f}')

    return df_flu


# ============ Part 2: Final Thesis Dataset ============
def build_extended_final_dataset():
    """组装 2019-11-01 ~ 2020-03-15 的环境 + 搜索指数数据"""
    print('\n=== Part 2: Building extended Final_Thesis_Dataset ===')

    date_range = pd.date_range(START_DATE, END_DATE)

    # 2a. Load NOAA weather (primary source for temp/humidity/precip/wind/pressure)
    df_noaa = pd.read_excel(os.path.join(NEW, 'NOAA_Weather_24Cities.xlsx'))
    df_noaa['date'] = pd.to_datetime(df_noaa['date'])
    # Normalize city names: add '市' if missing
    df_noaa['city_std'] = df_noaa['城市'].apply(lambda x: x if x.endswith('市') else x + '市')

    # 2b. Load AQI data
    df_aqi = pd.read_excel(os.path.join(NEW, '08_AQI_201911_202003.xlsx'))
    df_aqi['date'] = pd.to_datetime(df_aqi['日期'])
    df_aqi['city_std'] = df_aqi['城市'].apply(lambda x: x if x.endswith('市') else x + '市')

    # 2c. Load weather condition text (from tianqi lishi)
    df_wc = pd.read_excel(os.path.join(NEW, '07_Weather_201911_202003.xlsx'))
    df_wc['date'] = pd.to_datetime(df_wc['日期'].str.replace('年', '-')
                                    .str.replace('月', '-').str.replace('日', ''))
    df_wc['city_std'] = df_wc['城市'].apply(lambda x: x if x.endswith('市') else x + '市')

    # 2d. Load bus/metro search indices (matrix format: city×date)
    # These only cover 2019-11-01 to 2019-12-31, 22 cities
    bus_matrix = pd.read_excel(os.path.join(NEW, '公交.xlsx'))
    metro_matrix = pd.read_excel(os.path.join(NEW, '地铁.xlsx'))

    # 2e. Also load existing Final_Thesis_Dataset for the 2020 period
    df_old = pd.read_excel(os.path.join(SRC, 'Final_Thesis_Dataset_2020.xlsx'))
    df_old['Date'] = pd.to_datetime(df_old['日期'])
    df_old['City_std'] = df_old['城市'].apply(lambda x: x if x.endswith('市') else x + '市')

    # Build the extended dataset
    rows = []
    for dt in date_range:
        for city in TARGET_CITIES:
            row = {'城市': city, '日期': dt}

            # NOAA weather
            noaa_row = df_noaa[(df_noaa['city_std'] == city) & (df_noaa['date'] == dt)]
            if len(noaa_row) > 0:
                nr = noaa_row.iloc[0]
                row['最终气温(℃)'] = nr['气温(℃)']
                row['相对湿度(%)'] = nr['相对湿度(%)']
                row['降水量(mm)'] = nr['降水量(mm)']
                row['平均风速(m/s)'] = nr['平均风速(m/s)']
                row['海平面气压(hPa)'] = nr['海平面气压(hPa)']
                row['气温来源'] = 'NOAA_GSOD'
            else:
                # Fallback to existing data if available
                old_row = df_old[(df_old['City_std']== city) & (df_old['Date'] == dt)]
                if len(old_row) > 0:
                    or_ = old_row.iloc[0]
                    row['最终气温(℃)'] = or_['最终气温(℃)']
                    row['相对湿度(%)'] = or_['相对湿度(%)'] if pd.notna(or_['相对湿度(%)']) else np.nan
                    row['降水量(mm)'] = or_['降水量(mm)'] if pd.notna(or_['降水量(mm)']) else np.nan
                    row['平均风速(m/s)'] = or_['平均风速(m/s)'] if pd.notna(or_['平均风速(m/s)']) else np.nan
                    row['海平面气压(hPa)'] = or_['海平面气压(hPa)'] if pd.notna(or_['海平面气压(hPa)']) else np.nan
                    row['气温来源'] = or_['气温来源']
                else:
                    for k in ['最终气温(℃)', '相对湿度(%)', '降水量(mm)', '平均风速(m/s)',
                               '海平面气压(hPa)']:
                        row[k] = np.nan
                    row['气温来源'] = np.nan

            # AQI data
            aqi_row = df_aqi[(df_aqi['city_std'] == city) & (df_aqi['date'] == dt)]
            if len(aqi_row) > 0:
                ar = aqi_row.iloc[0]
                row['质量等级'] = ar['质量等级']
                row['PM2.5'] = ar['PM2.5']
                row['PM10'] = ar['PM10']
            else:
                old_row = df_old[(df_old['City_std']== city) & (df_old['Date'] == dt)]
                if len(old_row) > 0:
                    row['质量等级'] = old_row.iloc[0]['质量等级']
                    row['PM2.5'] = old_row.iloc[0]['PM2.5']
                    row['PM10'] = old_row.iloc[0]['PM10']
                else:
                    row['质量等级'] = np.nan
                    row['PM2.5'] = np.nan
                    row['PM10'] = np.nan

            # Weather condition text
            wc_row = df_wc[(df_wc['city_std'] == city) & (df_wc['date'] == dt)]
            if len(wc_row) > 0:
                row['天气状况(白天/夜间)'] = wc_row.iloc[0]['天气状况(白天/夜间)']
            else:
                old_row = df_old[(df_old['City_std']== city) & (df_old['Date'] == dt)]
                if len(old_row) > 0:
                    row['天气状况(白天/夜间)'] = old_row.iloc[0]['天气状况(白天/夜间)']
                else:
                    row['天气状况(白天/夜间)'] = np.nan

            # Bus/metro search index
            # First try matrix format files (2019-11~12), then fallback to old data (2020)
            date_str = dt.strftime('%Y-%m-%d')
            bus_val = np.nan
            metro_val = np.nan

            # Try 2019 matrix data (city names with 市, 22 cities, missing 上海/天津)
            if date_str in bus_matrix.columns:
                city_row = bus_matrix[bus_matrix.iloc[:, 0] == city]
                if len(city_row) > 0:
                    bus_val = city_row[date_str].values[0]
                city_row_m = metro_matrix[metro_matrix.iloc[:, 0] == city]
                if len(city_row_m) > 0:
                    metro_val = city_row_m[date_str].values[0]

            # Fallback to old data
            if pd.isna(bus_val) or pd.isna(metro_val):
                old_row = df_old[(df_old['City_std']== city) & (df_old['Date'] == dt)]
                if len(old_row) > 0:
                    if pd.isna(bus_val):
                        bus_val = old_row.iloc[0]['公交百度搜索指数']
                    if pd.isna(metro_val):
                        metro_val = old_row.iloc[0]['地铁百度搜索指数']

            row['公交百度搜索指数'] = bus_val if not pd.isna(bus_val) else 0
            row['地铁百度搜索指数'] = metro_val if not pd.isna(metro_val) else 0

            rows.append(row)

    df_final = pd.DataFrame(rows)

    # Add Unnamed: 12 column (always NaN)
    df_final['Unnamed: 12'] = np.nan

    # Ensure column order matches original
    col_order = ['城市', '日期', '最终气温(℃)', '相对湿度(%)', '降水量(mm)',
                 '平均风速(m/s)', '海平面气压(hPa)', '质量等级', 'PM2.5', 'PM10',
                 '气温来源', '天气状况(白天/夜间)', 'Unnamed: 12',
                 '地铁百度搜索指数', '公交百度搜索指数']
    df_final = df_final[col_order]

    # Save
    out_path = os.path.join(OUT, 'Final_Thesis_Dataset_extended.xlsx')
    df_final.to_excel(out_path, index=False)
    print(f'  Saved: {out_path}')
    print(f'  Shape: {df_final.shape}')
    print(f'  Date range: {df_final["日期"].min()} ~ {df_final["日期"].max()}')

    # Completeness report
    for col in ['最终气温(℃)', '相对湿度(%)', '降水量(mm)', '平均风速(m/s)',
                '海平面气压(hPa)', '质量等级', 'PM2.5', 'PM10',
                '公交百度搜索指数', '地铁百度搜索指数']:
        nn = df_final[col].notna().sum()
        print(f'  {col}: {nn}/{len(df_final)} ({100*nn/len(df_final):.0f}%)')

    return df_final


if __name__ == '__main__':
    df_flu = build_extended_flu()
    df_final = build_extended_final_dataset()
    print('\n=== Done! ===')
    print('Output files in data/新增数据/:')
    print('  01_Influenza_Target_extended.xlsx')
    print('  Final_Thesis_Dataset_extended.xlsx')
