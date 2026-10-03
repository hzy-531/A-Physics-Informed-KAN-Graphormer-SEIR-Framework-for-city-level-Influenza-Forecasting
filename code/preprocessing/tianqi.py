"""
历史气温爬虫 - V3.0 (最终修正版)
修复: 乱码问题 (GBK -> UTF-8)
"""
import requests
import pandas as pd
import time
import random
from io import StringIO

# ================= 配置区 =================
CITIES = {
    'beijing': '北京',      'tianjin': '天津',      'shanghai': '上海',     'chongqing': '重庆',
    'guangzhou': '广州',    'shenzhen': '深圳',     'xian': '西安',         'chengdu': '成都',
    'wuhan': '武汉',        'hangzhou': '杭州',     'nanjing': '南京',      'suzhou': '苏州',
    'wuxi': '无锡',         'zhengzhou': '郑州',    'changsha': '长沙',     'shenyang': '沈阳',
    'dalian': '大连',       'qingdao': '青岛',      'jinan': '济南',        'ningbo': '宁波',
    'xiamen': '厦门',       'haerbin': '哈尔滨',    'changchun': '长春',    'shijiazhuang': '石家庄'
}

MONTHS = ['201911', '201912', '202001', '202002', '202003']

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'http://www.tianqihoubao.com/',
}

def get_weather_data():
    all_data = []
    print(f"🚀 [V3.0] 开始重新爬取 (UTF-8模式)...")

    for pinyin, name in CITIES.items():
        print(f"\n正在处理: {name} ({pinyin})")

        for month in MONTHS:
            url = f"http://www.tianqihoubao.com/lishi/{pinyin}/month/{month}.html"

            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)

                # 🔥 关键修改：强制使用 UTF-8 编码 🔥
                resp.encoding = 'utf-8'

                # 解析表格
                dfs = pd.read_html(StringIO(resp.text), header=0)

                if dfs:
                    df = dfs[0]

                    # 清洗数据
                    if '日期' in df.columns:
                        df = df[df['日期'] != '日期'] # 去除重复表头
                        # 去除日期里的换行符和空格
                        df['日期'] = df['日期'].astype(str).str.replace(r'\s+', '', regex=True)

                    df['城市'] = name
                    all_data.append(df)
                    print(f"  ✅ {month} 获取成功")
                else:
                    print(f"  ⚠️ {month} 未找到表格")

                time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                print(f"  ❌ {month} 失败: {e}")

    if all_data:
        print("\n正在合并数据...")
        final_df = pd.concat(all_data, ignore_index=True)

        # 保存为新文件，避免混淆
        output_file = r'E:\Claude code\KAN+\data\新增数据\07_Weather_201911_202003.xlsx'
        final_df.to_excel(output_file, index=False)
        print(f"🎉 完美搞定！无乱码数据已保存至: {output_file}")
    else:
        print("😭 未获取到数据")

if __name__ == "__main__":
    get_weather_data()