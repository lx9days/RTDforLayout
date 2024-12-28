
# create synthetic point clouds
import numpy as np

print(123)
np.random.seed(7)
P = np.random.rand(1000, 2)
Q = np.random.rand(1000, 2)
# 
# calculate and plot R-Cross-Barcode
#
import rtd
barc = rtd.calc_embed_dist(P, Q)
rtd.plot_barcodes(rtd.barc2array(barc))
# import torch
# print(torch.cuda.is_available())

#
# Calculate RTD
#
# print(rtd.rtd(P, Q))