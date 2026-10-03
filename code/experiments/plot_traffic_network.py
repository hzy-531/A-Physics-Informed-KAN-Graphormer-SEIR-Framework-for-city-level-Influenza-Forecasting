"""
生成城市间交通流网络图 — 用于论文 Methods 部分
展示24城市间的迁移矩阵(OD矩阵)热力图和网络拓扑图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_lib.config import TARGET_CITIES, FILE_MIG_PATH
import networkx as nx
from matplotlib.lines import Line2D

def load_mobility_data():
    """加载迁徙数据"""
    xls = pd.ExcelFile(FILE_MIG_PATH)
    sheet_names = sorted([name for name in xls.sheet_names if name.startswith('D')])
    cities = TARGET_CITIES

    # 取所有有数据的日期，计算平均迁移矩阵
    adj_mats = []
    for sheet in sheet_names:
        df_mat = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        df_mat = df_mat.reindex(index=cities, columns=cities, fill_value=0.0)
        row_sums = df_mat.sum(axis=1)
        row_sums[row_sums == 0] = 1
        df_mat_norm = df_mat.div(row_sums, axis=0)
        adj_mats.append(df_mat_norm.values)

    avg_adj = np.mean(adj_mats, axis=0)
    return avg_adj, cities

def plot_traffic_network(adj_matrix, cities, output_dir):
    """绘制城市间交通流网络图（双面板：热力图 + 网络图）"""
    os.makedirs(output_dir, exist_ok=True)

    # 城市简称
    short_names = {
        '北京市': 'Beijing', '天津市': 'Tianjin', '石家庄市': 'Shijiazhuang',
        '沈阳市': 'Shenyang', '哈尔滨市': 'Harbin', '上海市': 'Shanghai',
        '南京市': 'Nanjing', '杭州市': 'Hangzhou', '合肥市': 'Hefei',
        '福州市': 'Fuzhou', '济南市': 'Jinan', '郑州市': 'Zhengzhou',
        '武汉市': 'Wuhan', '长沙市': 'Changsha', '广州市': 'Guangzhou',
        '南宁市': 'Nanning', '海口市': 'Haikou', '成都市': 'Chengdu',
        '贵阳市': 'Guiyang', '昆明市': 'Kunming', '西安市': "Xi'an",
        '兰州市': 'Lanzhou', '西宁市': 'Xining', '银川市': 'Yinchuan'
    }
    city_labels = [short_names.get(c, c[:3]) for c in cities]

    # 区域分组
    region_groups = {
        'North': ['北京市', '天津市', '石家庄市', '沈阳市', '哈尔滨市', '济南市', '郑州市'],
        'East': ['上海市', '南京市', '杭州市', '合肥市', '福州市'],
        'Central': ['武汉市', '长沙市'],
        'South': ['广州市', '南宁市', '海口市'],
        'West': ['成都市', '贵阳市', '昆明市', '西安市', '兰州市', '西宁市', '银川市']
    }
    region_colors = {
        'North': '#E74C3C', 'East': '#3498DB', 'Central': '#2ECC71',
        'South': '#F39C12', 'West': '#9B59B6'
    }

    city_to_region = {}
    for region, city_list in region_groups.items():
        for city in city_list:
            city_to_region[city] = region

    # 按区域重排城市
    sorted_cities = []
    sorted_regions = []
    for region in ['North', 'East', 'Central', 'South', 'West']:
        for city in cities:
            if city_to_region.get(city) == region:
                sorted_cities.append(city)
                sorted_regions.append(region)

    # 重排邻接矩阵
    idx_map = {c: i for i, c in enumerate(cities)}
    sorted_idx = [idx_map[c] for c in sorted_cities]
    adj_sorted = adj_matrix[np.ix_(sorted_idx, sorted_idx)]
    sorted_labels = [short_names.get(c, c[:3]) for c in sorted_cities]

    fig = plt.figure(figsize=(20, 9))

    # ========== 左面板：OD矩阵热力图 ==========
    ax1 = fig.add_subplot(1, 2, 1)

    # 对数变换以更好显示差异
    adj_log = np.log1p(adj_sorted * 100)  # 放大后取log

    im = ax1.imshow(adj_log, cmap='YlOrRd', aspect='auto', interpolation='bilinear')

    # 标注区域分隔线
    region_boundaries = []
    current_region = None
    for i, r in enumerate(sorted_regions):
        if r != current_region:
            region_boundaries.append(i - 0.5)
            current_region = r
    region_boundaries.append(len(sorted_cities) - 0.5)

    for boundary in region_boundaries[1:-1]:
        ax1.axhline(y=boundary, color='#34495E', linewidth=1.5, alpha=0.7)
        ax1.axvline(x=boundary, color='#34495E', linewidth=1.5, alpha=0.7)

    ax1.set_xticks(range(len(sorted_cities)))
    ax1.set_yticks(range(len(sorted_cities)))
    ax1.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax1.set_yticklabels(sorted_labels, fontsize=7)
    ax1.set_title('A  Inter-city Mobility Matrix (log-scale)', fontsize=12, fontweight='bold', loc='left')
    ax1.set_xlabel('Destination City', fontsize=10)
    ax1.set_ylabel('Origin City', fontsize=10)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('log(1 + 100 × Mobility Intensity)', fontsize=9)

    # 区域图例
    legend_elements = []
    for region in ['North', 'East', 'Central', 'South', 'West']:
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=region_colors[region],
                         alpha=0.6, label=region)
        )
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=7,
              bbox_to_anchor=(1.02, 1.0), title='Region', title_fontsize=8)

    # ========== 右面板：交通流网络图 ==========
    ax2 = fig.add_subplot(1, 2, 2)

    G = nx.DiGraph()

    # 城市坐标（按中国地理分布近似）
    geo_coords = {
        '北京市': (116.4, 39.9), '天津市': (117.2, 39.1), '石家庄市': (114.5, 38.0),
        '沈阳市': (123.4, 41.8), '哈尔滨市': (126.6, 45.8), '上海市': (121.5, 31.2),
        '南京市': (118.8, 32.1), '杭州市': (120.2, 30.3), '合肥市': (117.3, 31.8),
        '福州市': (119.3, 26.1), '济南市': (117.0, 36.7), '郑州市': (113.7, 34.8),
        '武汉市': (114.3, 30.6), '长沙市': (113.0, 28.2), '广州市': (113.3, 23.1),
        '南宁市': (108.3, 22.8), '海口市': (110.3, 20.0), '成都市': (104.1, 30.6),
        '贵阳市': (106.7, 26.6), '昆明市': (102.7, 25.0), '西安市': (108.9, 34.3),
        '兰州市': (103.8, 36.1), '西宁市': (101.8, 36.6), '银川市': (106.3, 38.5)
    }

    for city in sorted_cities:
        lon, lat = geo_coords.get(city, (110, 35))
        G.add_node(city, pos=(lon, lat), region=city_to_region.get(city, 'Other'))

    # 添加主要边（top 10% 强度的边）
    threshold = np.percentile(adj_matrix[adj_matrix > 0], 85)
    for i, src in enumerate(cities):
        for j, dst in enumerate(cities):
            if i != j and adj_matrix[i, j] > threshold:
                G.add_edge(src, dst, weight=adj_matrix[i, j])

    pos = {city: geo_coords.get(city, (110, 35)) for city in G.nodes()}

    node_colors = [region_colors.get(city_to_region.get(c, 'Other'), '#95A5A6')
                   for c in G.nodes()]
    node_sizes = [300 + 200 * (G.degree(c) / max(1, max(dict(G.degree()).values())))
                  for c in G.nodes()]

    # 绘制边
    edge_widths = []
    edge_alphas = []
    for u, v, d in G.edges(data=True):
        w = d['weight'] * 30
        edge_widths.append(w)
        edge_alphas.append(min(0.8, w * 0.5))

    nx.draw_networkx_edges(G, pos, ax=ax2, width=edge_widths, alpha=edge_alphas,
                          edge_color='#7F8C8D', arrows=True, arrowsize=8,
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, node_color=node_colors,
                          alpha=0.9, edgecolors='white', linewidths=1.5)

    # 标注主要城市
    major_cities = ['北京市', '上海市', '广州市', '武汉市', '成都市', '西安市',
                   '南京市', '杭州市', '深圳市' if '深圳市' in cities else '长沙市']
    label_pos = {c: (pos[c][0], pos[c][1] + 0.8) for c in G.nodes()}
    labels = {c: short_names.get(c, c) for c in G.nodes() if c in major_cities or G.degree(c) > 3}
    nx.draw_networkx_labels(G, label_pos, labels, ax=ax2, font_size=8, font_weight='bold')

    ax2.set_title('B  Inter-city Traffic Flow Network', fontsize=12, fontweight='bold', loc='left')
    ax2.set_xlim(100, 128)
    ax2.set_ylim(18, 48)
    ax2.axis('off')

    # 边权图例
    legend_edges = [
        Line2D([0], [0], color='#7F8C8D', linewidth=1, alpha=0.3, label='Low flow'),
        Line2D([0], [0], color='#7F8C8D', linewidth=3, alpha=0.6, label='Medium flow'),
        Line2D([0], [0], color='#7F8C8D', linewidth=6, alpha=0.8, label='High flow'),
    ]
    legend_nodes = [plt.Rectangle((0, 0), 1, 1, facecolor=region_colors[r],
                                  alpha=0.9, label=r) for r in ['North', 'East', 'Central', 'South', 'West']]
    leg1 = ax2.legend(handles=legend_edges, loc='lower left', fontsize=7, title='Flow Intensity')
    leg2 = ax2.legend(handles=legend_nodes, loc='upper right', fontsize=7, title='Region')
    ax2.add_artist(leg1)

    plt.suptitle('Inter-city Mobility Network Structure (24 Chinese Cities, 2019–2020)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, 'traffic_flow_network.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[完成] 交通流网络图已保存至: {out_path}")
    return out_path

def plot_degree_distribution(adj_matrix, cities, output_dir):
    """绘制迁移网络度分布图（补充图）"""
    os.makedirs(output_dir, exist_ok=True)

    in_degree = np.sum(adj_matrix, axis=0)
    out_degree = np.sum(adj_matrix, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    short_names = {
        '北京市': 'Beijing', '天津市': 'Tianjin', '石家庄市': 'Shijiazhuang',
        '沈阳市': 'Shenyang', '哈尔滨市': 'Harbin', '上海市': 'Shanghai',
        '南京市': 'Nanjing', '杭州市': 'Hangzhou', '合肥市': 'Hefei',
        '福州市': 'Fuzhou', '济南市': 'Jinan', '郑州市': 'Zhengzhou',
        '武汉市': 'Wuhan', '长沙市': 'Changsha', '广州市': 'Guangzhou',
        '南宁市': 'Nanning', '海口市': 'Haikou', '成都市': 'Chengdu',
        '贵阳市': 'Guiyang', '昆明市': 'Kunming', '西安市': "Xi'an",
        '兰州市': 'Lanzhou', '西宁市': 'Xining', '银川市': 'Yinchuan'
    }
    city_labels = [short_names.get(c, c) for c in cities]

    # 按流入度排序
    sorted_idx = np.argsort(in_degree)[::-1]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cities)))

    axes[0].barh(range(len(cities)), in_degree[sorted_idx], color=colors, edgecolor='white')
    axes[0].set_yticks(range(len(cities)))
    axes[0].set_yticklabels([city_labels[i] for i in sorted_idx], fontsize=8)
    axes[0].set_xlabel('In-degree (weighted inflow)', fontsize=11)
    axes[0].set_title('City Inflow Intensity', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')

    axes[1].barh(range(len(cities)), out_degree[sorted_idx], color=colors, edgecolor='white')
    axes[1].set_yticks(range(len(cities)))
    axes[1].set_yticklabels([city_labels[i] for i in sorted_idx], fontsize=8)
    axes[1].set_xlabel('Out-degree (weighted outflow)', fontsize=11)
    axes[1].set_title('City Outflow Intensity', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('City-level Mobility Degree Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'mobility_degree_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[完成] 度分布图已保存至: {out_path}")
    return out_path


if __name__ == '__main__':
    print(">>> 生成城市间交通流网络图...")
    adj_matrix, cities = load_mobility_data()
    print(f"  邻接矩阵形状: {adj_matrix.shape}")
    print(f"  城市数: {len(cities)}")
    print(f"  平均连接强度: {adj_matrix.mean():.4f}")
    print(f"  非零边比例: {(adj_matrix > 0).sum() / (adj_matrix.size - len(cities)):.2%}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'result', 'traffic_network')

    plot_traffic_network(adj_matrix, cities, output_dir)
    plot_degree_distribution(adj_matrix, cities, output_dir)
    print(">>> 完成！")
