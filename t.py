
# create synthetic point clouds
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import rtd1
from util import load_data, draw_Graph

# 设置文件夹路径和文件名
folder_path = './data'
edges_filename = 't-edge.mtx'
coords_filename = 't-pos.txt'
distance_filename = 't-origin.txt'

# folder_path = './data'
# edges_filename = 'ca-CSphd.mtx'
# coords_filename = 'ca-CSphd.txt'
# distance_filename = 'tmp_sp.txt'

# 调用函数
G, coordinates, dist_layout, dist_origin = load_data(folder_path, edges_filename, coords_filename, distance_filename)

# draw_Graph(G, coordinates)


# 
# calculate and plot R-Cross-Barcode
#

dist_quantile_origin, dist_quantile_layout = rtd1.get_embed_dist(dist_origin, dist_layout)

# print("Maximum value in EE:", np.max(dist_quantile_origin))
# print(barc)
print(dist_origin)
print(dist_layout)
barc = rtd1.calc_embed_dist(dist_origin, dist_layout, fast=True)
print(barc)
# for i, (birth, death) in enumerate(barc[1]):
#     print(f"Row {i}: BirthEdgeNode = {birth}, {death}")
rtd1.plot_barcodes_with_alpha(rtd1.barc2array(barc), 0.10, G, coordinates, dist_quantile_layout)

#
# Calculate RTD
#
# print(rtd1.rtd_dis(dist_origin, dist_layout))

