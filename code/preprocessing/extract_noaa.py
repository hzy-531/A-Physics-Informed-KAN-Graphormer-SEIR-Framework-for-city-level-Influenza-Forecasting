"""
从 NOAA GSOD tar.gz 提取24城市气象数据
湿度/降水/风速/气压 → 补充 Final_Thesis_Dataset 缺口
"""
import tarfile
import pandas as pd
import numpy as np
import csv
import io
from datetime import datetime
from collections import defaultdict

# ================= 24城市坐标 (纬度, 经度) =================
TARGET_CITIES = {
    '北京': (39.90, 116.40),
    '天津': (39.13, 117.20),
    '上海': (31.23, 121.47),
    '重庆': (29.56, 106.55),
    '广州': (23.13, 113.26),
    '深圳': (22.54, 114.06),
    '成都': (30.57, 104.07),
    '武汉': (30.59, 114.31),
    '杭州': (30.27, 120.15),
    '南京': (32.06, 118.80),
    '苏州': (31.30, 120.62),
    '无锡': (31.57, 120.30),
    '西安': (34.26, 108.94),
    '郑州': (34.75, 113.66),
    '长沙': (28.23, 112.94),
    '沈阳': (41.80, 123.43),
    '大连': (38.91, 121.61),
    '青岛': (36.07, 120.38),
    '济南': (36.65, 117.00),
    '宁波': (29.87, 121.54),
    '厦门': (24.48, 118.09),
    '哈尔滨': (45.75, 126.63),
    '长春': (43.90, 125.22),
    '石家庄': (38.04, 114.51),
}

# Manual station mapping for well-known city-airport stations
# Priority: matching by known station ID
CITY_STATION_MAP = {
    '北京': '545110',      # BEIJING CAPITAL INTERNATIONAL AIRPORT
    '天津': '545270',      # TIANJIN
    '石家庄': '536980',    # SHIJIAZHUANG
    '济南': '548230',      # JINAN TSINAN
    '青岛': '548570',      # LIUTING
    '沈阳': '543420',      # SHENYANG
    '大连': '546620',      # ZHOUSHUIZI
    '长春': '541610',      # LONGJIA
    '哈尔滨': '509530',    # HARBIN
    '上海': '583620',      # SHANGHAI
    '南京': '582380',      # LUKOU
    '杭州': '584570',      # XIAOSHAN
    '宁波': '582390',      # LISHE
    '厦门': '591340',      # GAOQI
    '广州': '592870',      # BAIYUN INTERNATIONAL
    '深圳': '594930',      # BAOAN INTERNATIONAL
    '武汉': '574940',      # TIANHE
    '长沙': '576870',      # CHANGSHA
    '郑州': '570830',      # XINZHENG
    '西安': '570360',      # XIANYANG
    '成都': '562940',      # SHUANGLIU
    '重庆': '575160',      # JIANGBEI
    '苏州': '583670',      # HONGQIAO (closest to Suzhou/Wuxi)
    '无锡': '583670',      # HONGQIAO
}


def parse_gsod_line(line):
    """Parse a GSOD CSV line using csv module for proper quote handling."""
    reader = csv.reader(io.StringIO(line))
    parts = next(reader)
    if len(parts) < 25:
        return None

    def parse_f(val, missing_thresh=9990):
        """Parse a GSOD numeric value. Default missing threshold 9990 for
        values stored in tenths (9999.9), lower for inches (99.99), etc."""
        v = val.strip()
        if not v:
            return np.nan
        try:
            f = float(v)
            if f >= missing_thresh - 0.01:
                return np.nan
            return f
        except (ValueError, TypeError):
            return np.nan

    return {
        'station': parts[0],
        'date': parts[1],
        'lat': parse_f(parts[2]),
        'lon': parse_f(parts[3]),
        'name': parts[5],
        'temp_f': parse_f(parts[6]),         # missing = 9999.9
        'dewp_f': parse_f(parts[8]),         # missing = 9999.9
        'slp_hpa': parse_f(parts[10]),       # missing = 9999.9
        'stp_hpa': parse_f(parts[12]),       # missing = 9999.9
        'visib': parse_f(parts[14], 990),    # missing = 999.9
        'wdsp_knots': parse_f(parts[16], 990), # missing = 999.9
        'mxspd': parse_f(parts[18], 990),    # missing = 999.9
        'gust': parse_f(parts[19], 990),     # missing = 999.9
        'max_f': parse_f(parts[20]),         # missing = 9999.9
        'min_f': parse_f(parts[22]),         # missing = 9999.9
        'prcp_inches': parse_f(parts[24], 90), # missing = 99.99
    }


def f_to_c(f):
    """Fahrenheit to Celsius."""
    return (f - 32) * 5.0 / 9.0


def knots_to_ms(k):
    """Knots to m/s."""
    return k * 0.514444


def inches_to_mm(i):
    """Inches to mm."""
    return i * 25.4


def compute_rh(temp_c, dewp_c):
    """
    Compute relative humidity (%) from temperature and dew point.
    Magnus formula: RH = 100 * exp((17.625*D)/(243.04+D)) / exp((17.625*T)/(243.04+T))
    """
    if np.isnan(temp_c) or np.isnan(dewp_c):
        return np.nan
    a = 17.625
    b = 243.04
    num = np.exp(a * dewp_c / (b + dewp_c))
    den = np.exp(a * temp_c / (b + temp_c))
    rh = 100.0 * num / den
    return min(max(rh, 0), 100)


def extract_noaa_data():
    all_rows = []

    for tar_name in ['2019.tar.gz', '2020.tar.gz']:
        tar_path = rf'E:\Claude code\KAN+\data\新增数据\{tar_name}'
        print(f'Processing {tar_name}...')

        with tarfile.open(tar_path, 'r:gz') as tar:
            # Collect needed station IDs (match by first 6 digits)
            needed_ids = set(CITY_STATION_MAP.values())
            members = [m for m in tar.getmembers()
                       if m.name.endswith('.csv') and m.name[:6] in needed_ids]

            for member in members:
                station_id = member.name[:6]  # Extract 6-digit WMO ID
                f = tar.extractfile(member)
                content = f.read().decode('utf-8')
                lines = content.strip().split('\n')

                for line in lines[1:]:  # Skip header
                    if not line.strip():
                        continue
                    row = parse_gsod_line(line)
                    if row:
                        row['station_id'] = station_id
                        all_rows.append(row)

    print(f'Total rows extracted: {len(all_rows)}')
    return pd.DataFrame(all_rows)


def build_city_weather(df):
    """Convert raw GSOD to city-level daily weather."""
    # Filter to target date range
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= '2019-11-01') & (df['date'] <= '2020-03-15')
    df = df[mask].copy()
    print(f'Rows in date range: {len(df)}')

    # Compute converted values
    df['气温(℃)'] = df['temp_f'].apply(f_to_c)
    df['dewp_c'] = df['dewp_f'].apply(f_to_c)
    df['相对湿度(%)'] = df.apply(lambda r: compute_rh(r['气温(℃)'], r['dewp_c']), axis=1)
    df['海平面气压(hPa)'] = df['slp_hpa']
    df['降水量(mm)'] = df['prcp_inches'].apply(inches_to_mm)
    df['平均风速(m/s)'] = df['wdsp_knots'].apply(knots_to_ms)

    # Map station to city (handle shared stations)
    station_to_cities = defaultdict(list)
    for city, sid in CITY_STATION_MAP.items():
        station_to_cities[sid].append(city)

    df['城市'] = df['station_id'].map(lambda s: station_to_cities.get(s, ['Unknown'])[0])

    # Duplicate rows for shared stations (e.g. Suzhou/Wuxi both use 583670)
    extra_rows = []
    for sid, cities in station_to_cities.items():
        if len(cities) > 1:
            sub = df[df['station_id'] == sid].copy()
            for extra_city in cities[1:]:
                dup = sub.copy()
                dup['城市'] = extra_city
                extra_rows.append(dup)
    if extra_rows:
        df = pd.concat([df] + extra_rows, ignore_index=True)

    # Check which cities have data
    print(f'Cities found: {sorted(df["城市"].dropna().unique())}')
    missing_cities = set(TARGET_CITIES.keys()) - set(df['城市'].dropna().unique())
    print(f'Cities MISSING: {missing_cities}')

    # Aggregate: if multiple stations per city (e.g. Shanghai), take mean
    result = df.groupby(['城市', 'date']).agg({
        '气温(℃)': 'mean',
        '相对湿度(%)': 'mean',
        '海平面气压(hPa)': 'mean',
        '降水量(mm)': 'mean',
        '平均风速(m/s)': 'mean',
        'station_id': 'first',
    }).reset_index()

    # Add source label
    result['气温来源'] = 'NOAA_GSOD'

    # Round to reasonable precision
    for col in ['气温(℃)', '相对湿度(%)', '海平面气压(hPa)', '降水量(mm)', '平均风速(m/s)']:
        result[col] = result[col].round(1)

    return result


def main():
    print('=' * 60)
    print('Extracting NOAA GSOD data for 24 cities')
    print('=' * 60)

    raw = extract_noaa_data()
    if len(raw) == 0:
        print('ERROR: No data extracted! Check station IDs.')
        return

    city_df = build_city_weather(raw)

    # Save
    output_path = r'E:\Claude code\KAN+\data\新增数据\NOAA_Weather_24Cities.xlsx'
    city_df.to_excel(output_path, index=False)
    print(f'\nSaved: {output_path}')
    print(f'Shape: {city_df.shape}')
    print(f'Date range: {city_df["date"].min()} ~ {city_df["date"].max()}')
    print(f'Cities: {city_df["城市"].nunique()}')

    # Show sample
    print('\nSample (Beijing):')
    bj = city_df[city_df['城市'] == '北京'].head(5)
    print(bj.to_string())


if __name__ == '__main__':
    main()
