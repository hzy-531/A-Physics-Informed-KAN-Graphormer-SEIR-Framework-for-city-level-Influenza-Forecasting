"""
百度迁徙数据爬虫优化版 - 24个中心城市2020年春季数据
专为提取2020年1月10日-3月31日数据设计
包含2020年春节（1月25日）
修复内容：
1. 修复None类型迭代错误
2. 修复JSON序列化错误
3. 增强错误处理和数据验证
4. 保持与2024版代码一致的健壮结构
"""
import requests
import json
import re
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import signal
import random
import hashlib

# ==================== 配置日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baidu_migration_jan_mar_2020.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 全局变量，用于控制程序退出 ====================
stop_signal = False

def signal_handler(sig, frame):
    """处理Ctrl+C信号"""
    global stop_signal
    print("\n\n接收到中断信号，正在优雅退出...")
    stop_signal = True

# ==================== 配置类 ====================
class BaiduMigrationConfig:
    """配置类 - 专为2020年数据优化"""
    # API接口配置
    API_BASE = "http://huiyan.baidu.com/migration"

    # API端点映射
    API_ENDPOINTS = {
        'cityrank': 'cityrank.jsonp',  # 市级排名数据
        'provincerank': 'provincerank.jsonp',  # 省级排名数据
        'historycurve': 'historycurve.jsonp',  # 历史迁徙曲线
        'internalflow': 'internalflowhistory.jsonp',  # 城内出行强度
        'lastdate': 'lastdate.jsonp',  # 最新日期
    }

    # 丰富的User-Agent列表
    USER_AGENTS = [
        # Chrome
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',

        # Firefox
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',

        # Safari
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',

        # Edge
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',

        # 移动端
        'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Linux; Android 13; SM-S901B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36',
    ]

    # 请求头基础模板
    BASE_HEADERS = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'Referer': 'http://qianxi.baidu.com/',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
    }

    # 请求间隔(秒) - 针对2020年数据增加间隔
    REQUEST_INTERVAL = 2.0
    MAX_RETRIES = 5

    # 数据存储配置
    DATA_DIR = Path('./data_24cities_jan_mar_2020')
    BACKUP_DIR = Path('./backup_24cities_jan_mar_2020')
    FAILED_TASKS_FILE = DATA_DIR / 'failed_tasks.json'

    @classmethod
    def get_random_user_agent(cls):
        """获取随机User-Agent"""
        return random.choice(cls.USER_AGENTS)

    @classmethod
    def get_headers(cls):
        """获取随机请求头"""
        headers = cls.BASE_HEADERS.copy()
        headers['User-Agent'] = cls.get_random_user_agent()
        return headers

    @classmethod
    def setup_directories(cls):
        """创建必要的目录"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.BACKUP_DIR.mkdir(exist_ok=True)
        (cls.DATA_DIR / 'json').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'csv').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'logs').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'summary').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'merged').mkdir(exist_ok=True)
        (cls.DATA_DIR / 'failed_records').mkdir(exist_ok=True)

# ==================== 24个中心城市管理器 ====================
class RegionManager:
    """24个中心城市管理器"""
    def __init__(self):
        self.regions = {}
        self._setup_24_cities()

    def _setup_24_cities(self):
        """设置24个中心城市"""
        cities = {
            # 直辖市
            '110000': '北京市',
            '120000': '天津市',
            '310000': '上海市',
            '500000': '重庆市',

            # 省会及副省级城市
            '440100': '广州市',
            '440300': '深圳市',
            '610100': '西安市',
            '510100': '成都市',
            '420100': '武汉市',
            '330100': '杭州市',
            '320100': '南京市',
            '320500': '苏州市',
            '320200': '无锡市',
            '410100': '郑州市',
            '430100': '长沙市',
            '210100': '沈阳市',
            '210200': '大连市',
            '370200': '青岛市',
            '370100': '济南市',
            '330200': '宁波市',
            '350200': '厦门市',
            '230100': '哈尔滨市',
            '220100': '长春市',
            '130100': '石家庄市',
        }

        for code, name in cities.items():
            self.regions[code] = {'name': name, 'level': self._get_level(code)}

    def _get_level(self, code: str) -> str:
        """根据代码判断级别"""
        if code.endswith('0000'):
            return 'province'
        else:
            return 'city'

    def get_all_cities(self) -> List[str]:
        """获取所有城市ID"""
        return list(self.regions.keys())

    def get_name(self, city_id: str) -> str:
        """获取城市名称"""
        return self.regions.get(city_id, {}).get('name', f'城市_{city_id}')

    def get_level(self, city_id: str) -> str:
        """获取城市级别"""
        return self.regions.get(city_id, {}).get('level', 'city')

    def get_city_info(self, city_id: str) -> Dict:
        """获取城市完整信息"""
        return {
            'id': city_id,
            'name': self.get_name(city_id),
            'level': self.get_level(city_id)
        }

# ==================== 百度迁徙API客户端 ====================
class BaiduMigrationAPI:
    """百度迁徙API客户端（修复版）"""

    def __init__(self):
        self.config = BaiduMigrationConfig
        self.session = requests.Session()

        # 配置连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        # 初始化请求头
        self._update_headers()
        self.request_count = 0
        self.last_request_time = time.time()

        # 请求统计
        self.request_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'retries': 0
        }

    def _update_headers(self):
        """更新请求头（随机User-Agent）"""
        headers = self.config.get_headers()
        self.session.headers.update(headers)

    def _rate_limit(self):
        """速率限制，带随机延迟"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        # 基础间隔 + 随机延迟
        base_interval = self.config.REQUEST_INTERVAL
        random_delay = random.uniform(0.5, 1.5)
        target_interval = base_interval + random_delay

        if elapsed < target_interval:
            sleep_time = target_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

        # 每10次请求更换一次User-Agent
        if self.request_count % 10 == 0:
            self._update_headers()

    def _exponential_backoff(self, attempt: int) -> float:
        """指数退避策略"""
        base = 2
        max_wait = 60
        wait_time = min(base ** attempt + random.uniform(0, 1), max_wait)
        return wait_time

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """发送API请求（带智能重试机制）"""
        global stop_signal

        if stop_signal:
            return None

        self._rate_limit()

        url = f"{self.config.API_BASE}/{endpoint}"

        # 为请求创建唯一标识
        req_id = hashlib.md5(f"{endpoint}_{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:8]

        for attempt in range(self.config.MAX_RETRIES):
            if stop_signal:
                return None

            try:
                # 每次重试前都更换User-Agent
                if attempt > 0:
                    self._update_headers()

                logger.debug(f"[{req_id}] 尝试第{attempt+1}次请求: {url}")

                # 增加超时时间
                response = self.session.get(url, params=params, timeout=20)

                # 检查状态码
                if response.status_code == 200:
                    # 提取JSONP数据
                    json_match = re.search(r'\(({.*})\)', response.text)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        if data.get('errmsg') == 'SUCCESS':
                            logger.debug(f"[{req_id}] 请求成功")
                            self.request_stats['success'] += 1
                            return data.get('data')
                        else:
                            logger.debug(f"[{req_id}] API返回错误: {data.get('errmsg', '未知错误')}")
                            # 特定错误不需要重试
                            if data.get('errmsg') in ['PARAM_ERROR', 'NO_DATA', 'date is not valid']:
                                return None
                    else:
                        logger.debug(f"[{req_id}] 未找到有效JSON数据")
                elif response.status_code == 502:
                    logger.warning(f"[{req_id}] 服务器502错误，等待重试")
                    wait_time = self._exponential_backoff(attempt) * 2
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [403, 429]:
                    logger.warning(f"[{req_id}] 被服务器限制(状态码{response.status_code})，等待较长时间")
                    wait_time = self._exponential_backoff(attempt) * 3
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"[{req_id}] 请求返回非200状态码: {response.status_code}")

            except requests.exceptions.Timeout as e:
                logger.warning(f"[{req_id}] 请求超时: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    wait_time = self._exponential_backoff(attempt)
                    logger.info(f"[{req_id}] 超时，等待{wait_time:.1f}秒后重试")
                    time.sleep(wait_time)
                    continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[{req_id}] 连接错误: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    wait_time = self._exponential_backoff(attempt)
                    logger.info(f"[{req_id}] 连接错误，等待{wait_time:.1f}秒后重试")
                    time.sleep(wait_time)
                    continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{req_id}] 请求异常: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    wait_time = self._exponential_backoff(attempt)
                    logger.info(f"[{req_id}] 请求异常，等待{wait_time:.1f}秒后重试")
                    time.sleep(wait_time)
                    continue
            except json.JSONDecodeError as e:
                logger.warning(f"[{req_id}] JSON解析失败: {e}")
                return None
            except Exception as e:
                logger.warning(f"[{req_id}] 未知错误: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    wait_time = self._exponential_backoff(attempt)
                    logger.info(f"[{req_id}] 未知错误，等待{wait_time:.1f}秒后重试")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"[{req_id}] 已达到最大重试次数，放弃请求")
                    break

            # 如果不是上述特定异常，进行重试
            if attempt < self.config.MAX_RETRIES - 1:
                wait_time = self._exponential_backoff(attempt)
                logger.info(f"[{req_id}] 请求失败，第{attempt+1}次重试，等待{wait_time:.1f}秒")
                time.sleep(wait_time)
                self.request_stats['retries'] += 1

        self.request_stats['failed'] += 1
        return None

    def get_migration_rank(self, city_id: str, rank_level: str,
                          move_type: str, date: str) -> Optional[List]:
        """获取迁徙排名数据"""
        # 确定API端点
        endpoint = 'cityrank.jsonp' if rank_level == 'city' else 'provincerank.jsonp'

        # 根据城市ID确定区域级别
        region_level = 'city' if len(city_id) == 6 and not city_id.endswith('0000') else 'province'

        params = {
            'dt': region_level,
            'id': city_id,
            'type': move_type,
            'date': date,
        }

        data = self._make_request(endpoint, params)
        if data and 'list' in data:
            return data['list']
        return None

    def get_internal_flow(self, city_id: str, date: str) -> Optional[float]:
        """获取城内出行强度"""
        params = {
            'dt': 'city',
            'id': city_id,
            'date': date,
        }

        data = self._make_request('internalflowhistory.jsonp', params)

        if data and 'list' in data and date in data['list']:
            try:
                return float(data['list'][date])
            except (ValueError, TypeError):
                return None
        return None

    def get_request_stats(self):
        """获取请求统计"""
        return self.request_stats.copy()

# ==================== 数据存储管理器 ====================
class DataStorage:
    """数据存储管理器"""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or BaiduMigrationConfig.DATA_DIR

    def save_daily_data(self, city_id: str, city_name: str, date: str,
                       move_in_city_rank: list, move_out_city_rank: list,
                       move_in_province_rank: list, move_out_province_rank: list,
                       internal_flow: Optional[float]) -> bool:
        """保存单日数据"""
        try:
            # 确保所有排名数据都是列表（修复None类型错误）
            move_in_city_rank = move_in_city_rank or []
            move_out_city_rank = move_out_city_rank or []
            move_in_province_rank = move_in_province_rank or []
            move_out_province_rank = move_out_province_rank or []

            # 创建数据结构
            data = {
                'city_id': city_id,
                'city_name': city_name,
                'date': date,
                'timestamp': datetime.now().isoformat(),
                'move_in_city_rank': move_in_city_rank,
                'move_out_city_rank': move_out_city_rank,
                'move_in_province_rank': move_in_province_rank,
                'move_out_province_rank': move_out_province_rank,
                'internal_flow': internal_flow,
            }

            # 保存JSON
            json_file = self.data_dir / 'json' / f'{city_name}_{date}.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 保存CSV（排名数据扁平化）
            csv_data = []

            # 处理迁入城市排名
            for i, item in enumerate(move_in_city_rank):
                if isinstance(item, dict):
                    csv_data.append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'date': date,
                        'rank_type': 'move_in_city',
                        'rank': i + 1,
                        'target_city_name': item.get('city_name', ''),
                        'target_province_name': item.get('province_name', ''),
                        'value': item.get('value', 0),
                    })

            # 处理迁出城市排名
            for i, item in enumerate(move_out_city_rank):
                if isinstance(item, dict):
                    csv_data.append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'date': date,
                        'rank_type': 'move_out_city',
                        'rank': i + 1,
                        'target_city_name': item.get('city_name', ''),
                        'target_province_name': item.get('province_name', ''),
                        'value': item.get('value', 0),
                    })

            # 处理迁入省份排名
            for i, item in enumerate(move_in_province_rank):
                if isinstance(item, dict):
                    csv_data.append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'date': date,
                        'rank_type': 'move_in_province',
                        'rank': i + 1,
                        'target_province_name': item.get('province_name', ''),
                        'value': item.get('value', 0),
                    })

            # 处理迁出省份排名
            for i, item in enumerate(move_out_province_rank):
                if isinstance(item, dict):
                    csv_data.append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'date': date,
                        'rank_type': 'move_out_province',
                        'rank': i + 1,
                        'target_province_name': item.get('province_name', ''),
                        'value': item.get('value', 0),
                    })

            # 添加内部流量数据
            if internal_flow is not None:
                csv_data.append({
                    'city_id': city_id,
                    'city_name': city_name,
                    'date': date,
                    'rank_type': 'internal_flow',
                    'rank': 0,
                    'target_city_name': '',
                    'target_province_name': '',
                    'value': internal_flow,
                })

            if csv_data:
                df = pd.DataFrame(csv_data)
                csv_file = self.data_dir / 'csv' / f'{city_name}_{date}.csv'
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')

            return True

        except Exception as e:
            logger.error(f"保存数据失败 {city_name} {date}: {e}")
            return False

    def save_summary(self, city_name: str, dates: List[str],
                    success_count: int, failed_count: int):
        """保存爬取摘要"""
        summary = {
            'city_name': city_name,
            'total_dates': len(dates),
            'success_dates': success_count,
            'failed_dates': failed_count,
            'success_rate': success_count / len(dates) if dates else 0,
            'last_update': datetime.now().isoformat(),
        }

        summary_file = self.data_dir / 'summary' / f'{city_name}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def save_failed_task(self, city_id: str, city_name: str, date: str, error: str = ""):
        """保存失败任务记录"""
        failed_file = self.data_dir / 'failed_records' / 'failed_tasks.json'

        # 读取现有的失败记录
        failed_tasks = []
        if failed_file.exists():
            try:
                with open(failed_file, 'r', encoding='utf-8') as f:
                    failed_tasks = json.load(f)
            except:
                failed_tasks = []

        # 添加新记录
        task_record = {
            'city_id': city_id,
            'city_name': city_name,
            'date': date,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'retry_count': 0
        }

        # 检查是否已存在相同记录
        existing = False
        for task in failed_tasks:
            if task['city_id'] == city_id and task['date'] == date:
                task['error'] = error
                task['timestamp'] = datetime.now().isoformat()
                task['retry_count'] += 1
                existing = True
                break

        if not existing:
            failed_tasks.append(task_record)

        # 保存记录
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_tasks, f, ensure_ascii=False, indent=2)

    def get_failed_tasks(self) -> List[Dict]:
        """获取所有失败任务"""
        failed_file = self.data_dir / 'failed_records' / 'failed_tasks.json'

        if failed_file.exists():
            try:
                with open(failed_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def clear_failed_task(self, city_id: str, date: str):
        """清除已成功的失败任务记录"""
        failed_file = self.data_dir / 'failed_records' / 'failed_tasks.json'

        if not failed_file.exists():
            return

        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                failed_tasks = json.load(f)

            # 过滤掉已成功的任务
            new_tasks = [task for task in failed_tasks
                        if not (task['city_id'] == city_id and task['date'] == date)]

            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(new_tasks, f, ensure_ascii=False, indent=2)
        except:
            pass

# ==================== 2020年1-3月数据爬虫 ====================
class JanMar2020DataCrawler:
    """2020年1月10日-3月31日数据爬虫"""

    def __init__(self):
        self.config = BaiduMigrationConfig
        self.api = BaiduMigrationAPI()
        self.storage = DataStorage()
        self.region_mgr = RegionManager()

        # 固定时间段：2020年1月10日-3月31日
        self.start_date = '20200110'
        self.end_date = '20200331'

        # 计算天数
        start_dt = datetime.strptime(self.start_date, '%Y%m%d')
        end_dt = datetime.strptime(self.end_date, '%Y%m%d')
        self.days_count = (end_dt - start_dt).days + 1

        # 24个中心城市列表
        self.target_cities = self.region_mgr.get_all_cities()

        # 统计数据
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
            'end_time': None,
            'failed_tasks': [],
        }

    def generate_date_range(self) -> List[str]:
        """生成日期范围"""
        dates = []
        current = datetime.strptime(self.start_date, '%Y%m%d')
        end = datetime.strptime(self.end_date, '%Y%m%d')

        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        return dates

    def crawl_city_day(self, city_id: str, date: str) -> Dict:
        """爬取单个城市单日数据 - 增强修复版"""
        global stop_signal

        if stop_signal:
            return {'city_id': city_id, 'date': date, 'status': 'cancelled'}

        city_name = self.region_mgr.get_name(city_id)
        result = {
            'city_id': city_id,
            'city_name': city_name,
            'date': date,
            'status': 'failed',
            'data_available': False,
        }

        try:
            # 顺序获取各种数据（避免并发触发反爬）
            move_in_city = self.api.get_migration_rank(city_id, 'city', 'move_in', date)
            time.sleep(random.uniform(0.5, 1.0))

            move_out_city = self.api.get_migration_rank(city_id, 'city', 'move_out', date)
            time.sleep(random.uniform(0.5, 1.0))

            move_in_province = self.api.get_migration_rank(city_id, 'province', 'move_in', date)
            time.sleep(random.uniform(0.5, 1.0))

            move_out_province = self.api.get_migration_rank(city_id, 'province', 'move_out', date)
            time.sleep(random.uniform(0.5, 1.0))

            internal_flow = self.api.get_internal_flow(city_id, date)

            # ===== 核心修复：增强型数据检查，避免任何 None 迭代 =====
            def safe_check_data(data):
                """安全地检查数据是否为非空列表"""
                if data is None:
                    return False
                if not isinstance(data, list):
                    # 记录非列表类型的意外数据，便于调试
                    logger.debug(f"数据检查：接收到非列表类型，类型为 {type(data)}，值为 {repr(data)[:100]}")
                    return False
                return len(data) > 0

            # 分别检查每个数据源
            data_sources = [
                ('move_in_city', move_in_city),
                ('move_out_city', move_out_city),
                ('move_in_province', move_in_province),
                ('move_out_province', move_out_province)
            ]

            has_data = False
            for source_name, source_data in data_sources:
                if safe_check_data(source_data):
                    has_data = True
                    logger.debug(f"{city_name} {date} 的 {source_name} 包含有效数据，共 {len(source_data)} 条")
                    break  # 只要有一个有效数据源即可

            # ===== 根据检查结果处理 =====
            if has_data:
                # 保存数据 - 确保即使部分数据为None，也传递空列表给存储方法
                success = self.storage.save_daily_data(
                    city_id, city_name, date,
                    move_in_city if safe_check_data(move_in_city) else [],
                    move_out_city if safe_check_data(move_out_city) else [],
                    move_in_province if safe_check_data(move_in_province) else [],
                    move_out_province if safe_check_data(move_out_province) else [],
                    internal_flow
                )

                if success:
                    result['status'] = 'success'
                    result['data_available'] = True
                    self.stats['successful_requests'] += 1
                    # 清除失败记录（如果存在）
                    self.storage.clear_failed_task(city_id, date)
                    logger.info(f"✓ {city_name} {date} 数据保存成功")
                else:
                    result['status'] = 'save_failed'
                    # 记录失败任务
                    self.storage.save_failed_task(city_id, city_name, date, "数据保存失败")
                    self.stats['failed_tasks'].append({
                        'city_id': city_id,
                        'city_name': city_name,
                        'date': date,
                        'error': '数据保存失败'
                    })
                    logger.warning(f"✗ {city_name} {date} 数据保存失败")
            else:
                result['status'] = 'no_data'
                # 记录无数据的失败任务
                self.storage.save_failed_task(city_id, city_name, date, "无有效数据")
                self.stats['failed_tasks'].append({
                    'city_id': city_id,
                    'city_name': city_name,
                    'date': date,
                    'error': '无有效数据'
                })
                logger.info(f"○ {city_name} {date} 无有效数据")

        except Exception as e:
            # 捕获更具体的错误信息
            error_detail = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"爬取 {city_name} {date} 失败: {error_detail}")
            result['status'] = 'error'
            result['error'] = error_detail
            # 记录异常失败任务
            self.storage.save_failed_task(city_id, city_name, date, error_detail)
            self.stats['failed_tasks'].append({
                'city_id': city_id,
                'city_name': city_name,
                'date': date,
                'error': error_detail
            })

        self.stats['total_requests'] += 1
        if result['status'] != 'success':
            self.stats['failed_requests'] += 1

        return result

    def crawl_city(self, city_id: str, progress_callback=None) -> Dict:
        """爬取单个城市的所有日期数据"""
        global stop_signal

        dates = self.generate_date_range()
        city_name = self.region_mgr.get_name(city_id)
        logger.info(f"开始爬取 {city_name} 数据，共 {len(dates)} 天")

        results = {
            'city_id': city_id,
            'city_name': city_name,
            'total_dates': len(dates),
            'success_dates': 0,
            'failed_dates': 0,
            'details': [],
        }

        for i, date in enumerate(dates):
            if stop_signal:
                logger.info(f"爬取被中断，停止 {city_name}")
                results['status'] = 'interrupted'
                break

            # 进度回调
            if progress_callback:
                progress_callback(i, len(dates))

            # 爬取单日数据
            day_result = self.crawl_city_day(city_id, date)
            results['details'].append(day_result)

            if day_result.get('status') == 'success':
                results['success_dates'] += 1
            else:
                results['failed_dates'] += 1

            # 每5天显示一次进度
            if i % 5 == 0 and i > 0:
                logger.info(f"{city_name}: 已处理 {i+1}/{len(dates)} 天，成功 {results['success_dates']} 天")

            # 每处理3天，随机休息，模拟人类行为
            if i % 3 == 0 and i > 0:
                rest_time = random.uniform(2, 5)
                time.sleep(rest_time)

        # 保存摘要
        self.storage.save_summary(city_name, dates,
                                 results['success_dates'], results['failed_dates'])

        logger.info(f"完成 {city_name}: 成功 {results['success_dates']}/{len(dates)} 天")
        return results

    def crawl_all_cities(self, max_workers: int = 1) -> Dict:
        """爬取所有城市的数据 - 使用单线程更稳定"""
        global stop_signal

        dates = self.generate_date_range()
        total_tasks = len(self.target_cities) * len(dates)

        logger.info(f"开始2020年春季数据爬取（修复版）")
        logger.info(f"城市数量: {len(self.target_cities)}")
        logger.info(f"日期范围: {self.start_date} 到 {self.end_date} ({len(dates)} 天)")
        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"重要时间点: 春节 - 2020年1月25日")
        logger.info(f"★ 修复: 解决None类型迭代和JSON序列化错误")

        self.stats['start_time'] = datetime.now()
        all_results = {}

        try:
            # 使用单线程避免触发反爬
            for idx, city_id in enumerate(self.target_cities):
                if stop_signal:
                    break

                city_name = self.region_mgr.get_name(city_id)

                # 城市间增加较长的随机间隔
                city_gap = random.uniform(5, 10)
                logger.info(f"等待 {city_gap:.1f} 秒后开始爬取第{idx+1}个城市: {city_name}")
                time.sleep(city_gap)

                try:
                    city_result = self.crawl_city(city_id)
                    all_results[city_id] = city_result

                    # 显示API统计
                    api_stats = self.api.get_request_stats()
                    logger.info(f"当前请求统计: 成功{api_stats['success']}, 失败{api_stats['failed']}, 重试{api_stats['retries']}")

                except Exception as e:
                    logger.error(f"城市 {city_name} 爬取失败: {e}")

        except KeyboardInterrupt:
            logger.info("用户中断爬取")
        except Exception as e:
            logger.error(f"爬取过程发生错误: {e}")
        finally:
            self.stats['end_time'] = datetime.now()

            # 确保datetime对象转换为字符串
            self._save_final_summary(all_results, dates)

            # 显示最终的API统计
            api_stats = self.api.get_request_stats()
            logger.info(f"最终API请求统计:")
            logger.info(f"  总计: {api_stats['total']}")
            logger.info(f"  成功: {api_stats['success']}")
            logger.info(f"  失败: {api_stats['failed']}")
            logger.info(f"  重试: {api_stats['retries']}")

        return all_results

    def _save_final_summary(self, all_results: Dict, dates: List[str]):
        """保存最终摘要报告 - 修复JSON序列化错误"""
        total_success = 0
        total_failed = 0

        # 确保所有datetime对象都转换为字符串
        start_time_str = self.stats['start_time'].isoformat() if self.stats['start_time'] else None
        end_time_str = self.stats['end_time'].isoformat() if self.stats['end_time'] else None

        duration_seconds = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration_seconds = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        summary = {
            'season': 'jan_mar_2020',
            'time_range': '2020年1月10日-3月31日',
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_dates': len(dates),
            'total_cities': len(self.target_cities),
            'start_time': start_time_str,
            'end_time': end_time_str,
            'duration_seconds': duration_seconds,
            'cities': [],
            'statistics': {
                'total_requests': self.stats['total_requests'],
                'successful_requests': self.stats['successful_requests'],
                'failed_requests': self.stats['failed_requests'],
                'success_rate': self.stats['successful_requests'] / self.stats['total_requests']
                    if self.stats['total_requests'] > 0 else 0,
                'failed_tasks_count': len(self.stats['failed_tasks']),
            }
        }

        for city_id, result in all_results.items():
            city_summary = {
                'city_id': city_id,
                'city_name': self.region_mgr.get_name(city_id),
                'total_dates': result.get('total_dates', 0),
                'success_dates': result.get('success_dates', 0),
                'failed_dates': result.get('failed_dates', 0),
                'status': result.get('status', 'unknown'),
            }
            summary['cities'].append(city_summary)

            total_success += result.get('success_dates', 0)
            total_failed += result.get('failed_dates', 0)

        summary['total_success_dates'] = total_success
        summary['total_failed_dates'] = total_failed

        # 保存摘要
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.config.DATA_DIR / f'jan_mar_2020_summary_{timestamp}.json'

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印最终统计
        print("\n" + "=" * 60)
        print("2020年1月10日-3月31日数据爬取完成！")
        print("=" * 60)
        print(f"城市数量: {len(self.target_cities)}")
        print(f"日期范围: {self.start_date} 到 {self.end_date} ({len(dates)} 天)")
        print(f"成功数据: {total_success} 天")
        print(f"失败数据: {total_failed} 天")
        print(f"总请求数: {self.stats['total_requests']}")
        print(f"成功率: {summary['statistics']['success_rate']:.1%}")

        if self.stats['failed_tasks']:
            print(f"失败任务数: {len(self.stats['failed_tasks'])}")
            print(f"失败记录保存在: {self.config.DATA_DIR / 'failed_records'}")

        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"总耗时: {duration}")

        print(f"\n数据保存位置: {self.config.DATA_DIR}")
        print(f"摘要文件: {summary_file}")
        print("=" * 60)

# ==================== 主函数 ====================
def crawl_jan_mar_2020():
    """主函数：爬取24个中心城市的2020年1月10日-3月31日数据"""
    print("=" * 70)
    print("百度迁徙数据爬虫 - 24城市2020年春季数据（修复版）")
    print("修复内容: 1. None类型迭代错误 2. JSON序列化错误")
    print("=" * 70)

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    # 初始化配置
    BaiduMigrationConfig.setup_directories()

    crawler = JanMar2020DataCrawler()
    region_mgr = RegionManager()

    print("\n24个中心城市列表:")
    for i, city_id in enumerate(crawler.target_cities, 1):
        city_name = region_mgr.get_name(city_id)
        print(f"{i:2d}. {city_name} ({city_id})")

    # 显示重要时间节点
    dates = crawler.generate_date_range()
    print(f"\n时间范围: {crawler.start_date} - {crawler.end_date} ({len(dates)} 天)")
    print("包含重要时间点:")
    print("  - 春节前返乡潮 (1月中下旬)")
    print("  ✨ 春节 (2020年1月25日)")
    print("  - 春节假期 (1月24日-30日)")
    print("  - 节后返工潮 (2月上旬)")
    print("  - 3月全国疫情逐步控制期")

    # 检查是否有失败的爬取任务
    failed_tasks = crawler.storage.get_failed_tasks()
    if failed_tasks:
        print(f"\n⚠️  检测到 {len(failed_tasks)} 个失败的爬取任务")
        retry = input("是否先重试失败的爬取任务？(y/n): ").strip().lower()
        if retry in ['y', 'yes']:
            print("注意：由于是历史数据，部分日期可能确实无数据")

    # 确认开始
    print("\n" + "=" * 70)
    print(f"爬取计划:")
    print(f"  城市数量: {len(crawler.target_cities)}")
    print(f"  时间范围: {crawler.start_date} 到 {crawler.end_date} ({len(dates)} 天)")

    # 计算预计请求数
    estimated_requests = len(crawler.target_cities) * len(dates) * 5
    estimated_time = estimated_requests * 2.0 / 3600  # 按2秒/请求计算小时

    print(f"  预计请求数: {estimated_requests:,}")
    print(f"  预计耗时: 约 {estimated_time:.1f} 小时 (视网络情况而定)")
    print(f"  数据保存位置: {BaiduMigrationConfig.DATA_DIR}")
    print("=" * 70)

    confirm = input("\n是否开始爬取2020年春季数据？(y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("爬取已取消")
        return

    print("\n开始爬取数据...")
    print("提示: 按Ctrl+C可以中断爬取，已爬取的数据会被保存")
    print("优化措施:")
    print("  ✓ 修复None类型迭代错误")
    print("  ✓ 修复JSON序列化错误")
    print("  ✓ 增强数据验证和错误处理")
    print("  ✓ 保守的单线程策略确保稳定性")
    print("-" * 70)

    try:
        # 开始爬取（使用单线程确保稳定）
        results = crawler.crawl_all_cities(max_workers=1)

        print("\n" + "=" * 70)
        print("2020年春季数据爬取完成！")
        print("=" * 70)

        # 显示简要统计
        if results:
            success_cities = sum(1 for r in results.values() if r.get('success_dates', 0) > 0)
            total_success = sum(r.get('success_dates', 0) for r in results.values())
            total_dates = len(crawler.target_cities) * len(dates)

            print(f"成功爬取数据城市: {success_cities}/{len(results)}")
            print(f"成功爬取天数: {total_success}/{total_dates}")

            # 检查是否有失败任务
            failed_tasks = crawler.storage.get_failed_tasks()
            if failed_tasks:
                print(f"失败任务数: {len(failed_tasks)}")
                print("注：部分日期可能确实无数据，这是正常现象")

            print("\n📊 建议下一步操作:")
            print("1. 查看 data_24cities_jan_mar_2020/ 目录中的CSV文件")
            print("2. 检查 json/ 目录中的原始数据")
            print("3. 结合你的流感数据进行分析")

    except KeyboardInterrupt:
        print("\n\n爬取被用户中断")
        print("已爬取的数据已保存，可以在 data_24cities_jan_mar_2020 目录中找到")
    except Exception as e:
        print(f"\n爬取过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主入口函数"""
    print("=" * 70)
    print("百度迁徙数据爬虫 v5.0 - 24城市2020年春季数据专版（修复版）")
    print("=" * 70)
    print("功能特点:")
    print("  ✓ 专门针对24个中心城市")
    print("  ✓ 时间范围：2020年1月10日-3月31日（82天）")
    print("  ✓ 包含2020年春节（1月25日）")
    print("  ✓ 修复了None类型迭代错误")
    print("  ✓ 修复了JSON序列化错误")
    print("  ✓ 增强数据验证和错误处理")
    print("  ✓ 保守策略确保历史数据获取稳定性")
    print("=" * 70)

    try:
        crawl_jan_mar_2020()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

    try:
        input("\n按Enter键退出...")
    except (EOFError, KeyboardInterrupt):
        print("\n程序退出")

if __name__ == "__main__":
    main()