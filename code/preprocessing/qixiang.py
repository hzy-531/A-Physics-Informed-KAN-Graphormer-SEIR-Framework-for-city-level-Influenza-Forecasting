# -*- coding: utf-8 -*-
"""
空气质量数据爬虫 (AQI/PM2.5)
目标: www.tianqihoubao.com/aqi/
包含: AQI指数, 质量等级, PM2.5, PM10, SO2, CO 等
"""
import requests
import pandas as pd
import time
import random
from io import StringIO

# ================= 配置区 =================
# 24个中心城市拼音 (不需要改)
CITIES = {
    'beijing': '北京', 'tianjin': '天津', 'shanghai': '上海', 'chongqing': '重庆',
    'guangzhou': '广州', 'shenzhen': '深圳', 'chengdu': '成都', 'wuhan': '武汉',
    'hangzhou': '杭州', 'nanjing': '南京', 'suzhou': '苏州', 'wuxi': '无锡',
    'xian': '西安', 'zhengzhou': '郑州', 'changsha': '长沙', 'shenyang': '沈阳',
    'dalian': '大连', 'qingdao': '青岛', 'jinan': '济南', 'ningbo': '宁波',
    'xiamen': '厦门', 'haerbin': '哈尔滨', 'changchun': '长春', 'shijiazhuang': '石家庄'
}

# 目标月份
MONTHS = ['201911', '201912', '202001', '202002', '202003']

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'http://www.tianqihoubao.com/',
}


def get_aqi_data():
    all_data = []
    print(f"🚀 开始爬取 24 个城市的空气质量数据 (AQI)...")

    for pinyin, name in CITIES.items():
        print(f"\n正在处理: {name} ({pinyin})")

        for month in MONTHS:
            # 注意：AQI 的 URL 规则和历史气温不一样
            # 规则: /aqi/beijing-202001.html
            url = f"http://www.tianqihoubao.com/aqi/{pinyin}-{month}.html"

            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)

                # 依然强制 UTF-8 防止乱码
                resp.encoding = 'utf-8'

                # 解析表格
                dfs = pd.read_html(StringIO(resp.text), header=0)

                if dfs:
                    df = dfs[0]

                    # 清洗数据
                    # 1. AQI 表格有时候会包含重复的表头，去掉它
                    if '日期' in df.columns:
                        df = df[df['日期'] != '日期']
                        # 2. 清洗日期格式 (去除空格换行)
                        df['日期'] = df['日期'].astype(str).str.replace(r'\s+', '', regex=True)

                    df['城市'] = name
                    all_data.append(df)
                    print(f"  ✅ {month} AQI 获取成功")
                else:
                    print(f"  ⚠️ {month} 未找到表格")

                time.sleep(random.uniform(0.5, 1.0))

            except Exception as e:
                print(f"  ❌ {month} 失败: {e}")

    # ================= 保存结果 =================
    if all_data:
        print("\n正在合并数据...")
        final_df = pd.concat(all_data, ignore_index=True)

        # 保存文件
        output_file = r'E:\Claude code\KAN+\data\新增数据\08_AQI_201911_202003.xlsx'
        final_df.to_excel(output_file, index=False)
        print(f"🎉 成功！空气质量数据已保存至: {output_file}")
        print("💡 数据包含: 日期, AQI, 质量等级, PM2.5, PM10, NO2, SO2, CO, O3")
    else:
        print("😭 未获取到数据。")


if __name__ == "__main__":
    get_aqi_data()