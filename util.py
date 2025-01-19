#
# create synthetic point clouds
#
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# 计算距离矩阵
def compute_distance_matrix(coords):
    # 获取节点数量
    n = coords.shape[0]

    # 初始化距离矩阵
    distance_matrix = np.zeros((n, n))

    # 计算每一对节点之间的欧几里得距离
    for i in range(n):
        for j in range(i+1, n):  # 只计算上三角部分，减少计算量
            dist = np.linalg.norm(coords[i] - coords[j])  # 欧几里得距离
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # 对称矩阵

    return distance_matrix


def load_data(folder_path, edges_filename, coords_filename, distance_filename):
    # 文件路径
    edges_file = f'{folder_path}/{edges_filename}'
    coords_file = f'{folder_path}/{coords_filename}'
    filename = f'{folder_path}/{distance_filename}'

    # 创建无向图
    G = nx.Graph()

    # 添加边到图中
    edges = np.loadtxt(edges_file, dtype=int, ndmin=2)
    G.add_edges_from([(u, v, {'color': 'gray'}) for u, v in edges])

    # 计算距离矩阵
    coordinates = np.loadtxt(coords_file)
    distance_matrix_layout = compute_distance_matrix(coordinates)

    # 读取图论距离
    distance_matrix_origin = np.loadtxt(filename)

    # 返回图结构和计算结果
    return G, coordinates, distance_matrix_layout, distance_matrix_origin


def draw_Graph(G, coordinates, ax):
    # 为每个节点添加坐标
    node_pos = {i: coordinates[i] for i in range(len(coordinates))}
    # 绘制图形
    ax.set_title("Graph Visualization")
    nx.draw(G, pos=node_pos, ax=ax, with_labels=False, node_size=50, edge_color='gray', alpha=0.5)




