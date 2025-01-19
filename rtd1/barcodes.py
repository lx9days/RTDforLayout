from sklearn.metrics.pairwise import pairwise_distances
from copy import copy, deepcopy
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import ripserplusplus as rpp_py
import networkx as nx

def pdist_gpu(a, b, device = 'cuda:0'):
    A = torch.tensor(a, dtype = torch.float64)
    B = torch.tensor(b, dtype = torch.float64)

    size = (A.shape[0] + B.shape[0]) * A.shape[1] / 1e9
    max_size = 0.2

    if size > max_size:
        parts = int(size / max_size) + 1
    else:
        parts = 1

    pdist = np.zeros((A.shape[0], B.shape[0]))
    At = A.to(device)

    for p in range(parts):
        i1 = int(p * B.shape[0] / parts)
        i2 = int((p + 1) * B.shape[0] / parts)
        i2 = min(i2, B.shape[0])

        Bt = B[i1:i2].to(device)
        pt = torch.cdist(At, Bt)
        pdist[:, i1:i2] = pt.cpu()

        del Bt, pt
        torch.cuda.empty_cache()

    del At

    return pdist

def sep_dist(a, b, pdist_device = 'cpu'):
    if pdist_device == 'cpu':
        d1 = pairwise_distances(b, a, n_jobs = 40)
        d2 = pairwise_distances(b, b, n_jobs = 40)
    else:
        d1 = pdist_gpu(b, a, device = pdist_device)
        d2 = pdist_gpu(b, b, device = pdist_device)

    s = a.shape[0] + b.shape[0]

    apr_d = np.zeros((s, s))
    apr_d[a.shape[0]:, :a.shape[0]] = d1
    apr_d[a.shape[0]:, a.shape[0]:] = d2

    return apr_d

def barc2array(barc):
    keys = sorted(barc.keys())
    
    arr = []
    
    for k in keys:
        res = np.zeros((len(barc[k]), 2))

        for idx in range(len(barc[k])):
            elem = barc[k][idx]
            res[idx, 0] = elem[0]
            res[idx, 1] = elem[1]
            
        arr.append(res)
        
    return arr
   
def get_embed_dist(a, b, inputType='distance', dim = 1, pdist_device = 'cuda:0', verbose = False, norm = 'quantile', metric = 'euclidean', use_max = False, fast = False):
    n = a.shape[0]
    
    if inputType == 'distance':
        r1 = a
        r2 = b
    else:
        if pdist_device == 'cpu':
            if verbose:
                print('pdist on cpu start')
            r1 = pairwise_distances(a, a, n_jobs = 40, metric = metric)
            r2 = pairwise_distances(b, b, n_jobs = 40, metric = metric)
        else:
            if verbose:
                print('pdist on gpu start')
            r1 = pdist_gpu(a, a, device = pdist_device)
            r2 = pdist_gpu(b, b, device = pdist_device)
    
    if norm == 'median':
        r1 = r1 / np.median(r1)
        r2 = r2 / np.median(r2)
    elif norm == 'quantile':
        r1 = r1 / np.quantile(r1, 0.9)
        r2 = r2 / np.quantile(r2, 0.9)
    elif norm == 'mean':
        r1 = r1 / np.mean(r1)
        r2 = r2 / np.mean(r2)
    else:
        raise ValueError('Unknown norm type')

    if verbose:
        print('pairwise distances calculated')
    
    
    
    return r1, r2

def calc_embed_dist(a, b, inputType='distance', dim = 1, pdist_device = 'cuda:0', verbose = False, norm = 'quantile', metric = 'euclidean', use_max = False, fast = False):
    print('RTD for LAYOUT')
    n = a.shape[0]
    
    if inputType == 'distance':
        r1 = a
        r2 = b
    else:
        if pdist_device == 'cpu':
            if verbose:
                print('pdist on cpu start')
            r1 = pairwise_distances(a, a, n_jobs = 40, metric = metric)
            r2 = pairwise_distances(b, b, n_jobs = 40, metric = metric)
        else:
            if verbose:
                print('pdist on gpu start')
            r1 = pdist_gpu(a, a, device = pdist_device)
            r2 = pdist_gpu(b, b, device = pdist_device)
    

    if norm == 'median':
        r1 = r1 / np.median(r1)
        r2 = r2 / np.median(r2)
    elif norm == 'quantile':
        r1 = r1 / np.quantile(r1, 0.9)
        r2 = r2 / np.quantile(r2, 0.9)
    elif norm == 'mean':
        r1 = r1 / np.mean(r1)
        r2 = r2 / np.mean(r2)
    else:
        raise ValueError('Unknown norm type')


    if verbose:
        print('pairwise distances calculated')


    #
    #  0      r1
    #  r1  min(r1,r2)
    #
    d = np.zeros((2 * n, 2 * n))
  
    if fast:
        r1_half = deepcopy(r1)
        r1_half[np.tril_indices(r1.shape[0], -1)] = np.inf
    else:
        r1_half = r1
    if not use_max:
        d[n:, :n] = r1_half
        d[:n, n:] = r1_half.T
        d[n:, n:] = np.minimum(r1, r2)
    else:
        d[n:, :n] = np.maximum(r1, r2)
        d[:n, n:] = np.maximum(r1, r2)
        d[n:, n:] = r2

    m = r1.mean()
    d[d < m*(1e-6)] = 0
    print(d)
    d_tril = d[np.tril_indices(d.shape[0], k = -1)]
    
    if verbose:
        print('matrix prepared')
    
    barc = rpp_py.run("--format lower-distance --dim %d" % dim, d_tril)
    
    return barc

def count_cross_barcodes(cloud_1, cloud_2, dim, title = '', cuda = 0, is_plot = False, pdist_device = 'cpu'):

    if pdist_device != 'cpu':
        pdist_device = 'cuda:%d' % cuda

    d = sep_dist(cloud_1, cloud_2, pdist_device = pdist_device)
    m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
    d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
    d[d < m*(1e-6)] = 0
    d_tril = d[np.tril_indices(d.shape[0], k = -1)]
    
    barcodes = rpp_py.run("--format lower-distance --dim %d" % dim, d_tril)

    if is_plot:
      plot_barcodes(barc2array(barcodes), title = title)
      plt.show()
    
    return barcodes
   
def plot_barcodes(arr, color_list = ['deepskyblue', 'limegreen', 'darkkhaki'], dark_color_list = None, title = '', hom = None, ax=None, fig=None):
    
    if ax is None:
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = [6, 4]
        show = True
    else:
        show = False

    if dark_color_list is None:
        dark_color_list = color_list
        #dark_color_list = ['b', 'g', 'orange']

    sh = len(arr)
    step = 0
    if (len(color_list) < sh):
        color_list *= sh

    for i in range(sh):

        if not (hom is None):
            if i not in hom:
                continue

        barc = arr[i].copy()
        arrayForSort = np.subtract(barc[:,1],barc[:,0])

        bsorted = np.sort(arrayForSort)
        nbarc = bsorted.shape[0]
        if show: print('H%d: num barcodes %d' % (i, nbarc))
        if nbarc:
            if show:
                print('max0,976Barcode',i,'=',bsorted[nbarc*976//1000])
                print('maxBarcode',i,'=',bsorted[-1])
                print('middleBarcode',i,'=',bsorted[nbarc//2])
            #print('minbarcode',i,'=',bsorted[0])
            max = bsorted[-3:]

            ax.plot(barc[0], np.ones(2)*step, color = color_list[i], label = 'H{}'.format(i))
            for b in barc:
                if b[1] - b[0] in max :
                    ax.plot(b, np.ones(2)*step, dark_color_list[i])
                else:
                    ax.plot(b, np.ones(2)*step, color = color_list[i])
                step += 1

    ax.set_xlabel('$\epsilon$ (time)')
    ax.set_ylabel('segment')
    ax.set_title(title)
    #ax.set_xlim(0, 1)
    ax.legend(loc = 'lower right')
    if show:
        plt.show()

def filter_edges(E, start, end):
    print('find range:', start, end)
    edges = []

    # 初始化最接近的边长值
    closest_start = None
    closest_end = None

    # 找到最接近的start和end值
    for i, j in zip(*np.triu_indices_from(E, k=1)):
        dist = round(E[i, j], 8)
        # 寻找最接近 start 的边
        if dist <= start and (closest_start is None or start - dist < start - closest_start):
            closest_start = dist
        # 寻找最接近 end 的边
        if dist >= end and (closest_end is None or dist - end < closest_end - end):
            closest_end = dist

    # 输出调试信息
    print('closest_start:', closest_start)
    print('closest_end:', closest_end)

    # 遍历矩阵，筛选边
    for i, j in zip(*np.triu_indices_from(E, k=1)):
        dist = round(E[i, j], 8)
        # 在范围内的边设置颜色
        if dist == closest_start:
            edges.append((i, j, {'color': 'green'}))  # 绿边
        elif dist == closest_end:
            edges.append((i, j, {'color': 'red'}))  # 红边

    print('total edges:', len(edges))


    return edges

def plot_barcodes_with_alpha(arr, alpha, G, coordinates, E, color_list=['deepskyblue', 'limegreen', 'darkkhaki'], dark_color_list=None, title='', hom=None, ax=None, fig=None):
    if ax is None or fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        show = True
    else:
        ax1, ax2 = ax
        show = False

    if dark_color_list is None:
        dark_color_list = color_list

    sh = len(arr)
    step = 0
    if len(color_list) < sh:
        color_list *= sh

    # Store the plotted barcodes for dynamic updates
    plotted_barcodes = []
    intersecting_barcodes = []
    barcode_positions = []

    for i in range(sh):
        if hom is not None and i not in hom:
            continue

        barc = arr[i].copy()
        barc = barc[::3]
        arrayForSort = np.subtract(barc[:, 1], barc[:, 0])

        bsorted = np.sort(arrayForSort)
        nbarc = bsorted.shape[0]
        if show: 
            print('H%d: num barcodes %d' % (i, nbarc))
        if nbarc:
            if show:
                print('max0,976Barcode', i, '=', bsorted[nbarc*976//1000])
                print('maxBarcode', i, '=', bsorted[-1])
                print('middleBarcode', i, '=', bsorted[nbarc//2])


            ax1.plot(barc[0], np.ones(2)*step, color=color_list[i], label='H{}'.format(i))
            for b in barc:
                line, = ax1.plot(b, np.ones(2)*step, color=color_list[i])
                plotted_barcodes.append((b, line))
                barcode_positions.append((b[0], b[1], step))
                step += 1

    # Add a vertical red line for alpha
    vline = ax1.axvline(x=alpha, color='red', linestyle='--', label='alpha={}'.format(alpha))

    def on_click(event):
        if event.inaxes == ax1:  # Update based on click
            vline.set_xdata([event.xdata])
            # Find the closest barcode to the click position
            clicked_barcode = None
            min_distance = float('inf')
            for b, line in plotted_barcodes:
                mid_point = (b[0] + b[1]) / 2
                distance = abs(event.xdata - mid_point)
                if distance < min_distance:
                    min_distance = distance
                    clicked_barcode = b

            if clicked_barcode is not None:
                # Highlight the selected barcode
                intersecting_barcodes.clear()
                for b, line in plotted_barcodes:
                    if np.array_equal(b, clicked_barcode):
                        line.set_color('orange')
                        intersecting_barcodes.append((float(b[0]), float(b[1])))


                # Remove existing green and red edges from the graph
                edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') in ['green', 'red']]
                G.remove_edges_from(edges_to_remove)

                # Dynamically update edges based on the selected barcode
                start = clicked_barcode[0]
                end = clicked_barcode[1]

                add_edges = filter_edges(E, start, end)
                print(add_edges)

                G.add_edges_from(add_edges)

                # Redraw the graph with updated edges
                ax2.clear()
                ax2.set_title("Graph Visualization")
                node_pos = {i: coordinates[i] for i in range(len(coordinates))}
                edge_colors = []
                edge_widths = []
                edge_alphas = []
                print(G.edges())
                for u, v in G.edges():
                    color = G[u][v]['color']
                    if color == 'green':
                        edge_colors.append('green')
                        edge_widths.append(2)
                        edge_alphas.append(1.0)
                    elif color == 'red':
                        edge_colors.append('red')
                        edge_widths.append(2)
                        edge_alphas.append(1.0)
                    else:
                        edge_colors.append('gray')
                        edge_widths.append(1)
                        edge_alphas.append(0.5)

                nx.draw(G, pos=node_pos, ax=ax2, with_labels=False, node_size=50, edge_color=edge_colors, width=edge_widths, alpha=0.8)

                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    ax1.set_xlabel('$\epsilon$ (time)')
    ax1.set_ylabel('segment')
    ax1.set_title(title)
    ax1.legend(loc='lower right')

    # Initial graph rendering
    node_pos = {i: coordinates[i] for i in range(len(coordinates))}
    ax2.set_title("Graph Visualization")
    edge_colors = [G[u][v]['color'] if 'color' in G[u][v] else 'gray' for u, v in G.edges()]
    nx.draw(G, pos=node_pos, ax=ax2, with_labels=False, node_size=50, edge_color=edge_colors, alpha=0.5)

    if show:
        plt.show()




def plot_barcodes_with_static_alpha(arr, alpha, color_list=['deepskyblue', 'limegreen', 'darkkhaki'], dark_color_list=None, title='', hom=None, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        plt.rcParams["figure.figsize"] = [6, 4]
        show = True
    else:
        show = False

    if dark_color_list is None:
        dark_color_list = color_list

    sh = len(arr)
    step = 0
    if len(color_list) < sh:
        color_list *= sh

    for i in range(sh):
        if hom is not None and i not in hom:
            continue

        barc = arr[i].copy()
        arrayForSort = np.subtract(barc[:, 1], barc[:, 0])

        bsorted = np.sort(arrayForSort)
        nbarc = bsorted.shape[0]
        if show: 
            print('H%d: num barcodes %d' % (i, nbarc))
        if nbarc:
            if show:
                print('max0,976Barcode', i, '=', bsorted[nbarc*976//1000])
                print('maxBarcode', i, '=', bsorted[-1])
                print('middleBarcode', i, '=', bsorted[nbarc//2])

            max = bsorted[-3:]

            ax.plot(barc[0], np.ones(2)*step, color=color_list[i], label='H{}'.format(i))
            for b in barc:
                if b[1] - b[0] in max:
                    ax.plot(b, np.ones(2)*step, dark_color_list[i])
                elif (b[0] <= alpha <= b[1]):  # Check for intersection with alpha
                    ax.plot(b, np.ones(2)*step, color='orange')  # Highlight in orange
                else:
                    ax.plot(b, np.ones(2)*step, color=color_list[i])
                step += 1

    # Add a vertical red line for alpha
    ax.axvline(x=alpha, color='red', linestyle='--', label='alpha={}'.format(alpha))

    ax.set_xlabel('$\epsilon$ (time)')
    ax.set_ylabel('segment')
    ax.set_title(title)
    ax.legend(loc='lower right')
    if show:
        plt.show()

def calc_barcodes(cloud, batch_size = -1, pdist_device = 'cuda:0', dim = 1, is_plot = False):
    dark_color_list = ['deepskyblue', 'maroon', 'g', 'darkorange']
    color_list = ['paleturquoise', 'lightcoral', 'lightgreen', 'lightsalmon']

    if batch_size == -1:
        cl = cloud
    else:
        batch_size = min(batch_size, cloud.shape[0])
        indexes = np.random.choice(cloud.shape[0], batch_size, replace=False)
        cl = cloud[indexes]
    
    if pdist_device == 'cpu':
        d = pairwise_distances(cl, cl, n_jobs = 40)
    else:
        d = pdist_gpu(cl, cl, device = pdist_device)

    m = d.mean()
    d[d < m*(1e-6)] = 0
    d_tril = d[np.tril_indices(d.shape[0], k = -1)]
    
    barcodes = rpp_py.run("--format lower-distance --dim %d" % dim, d_tril)

    if is_plot:
      plot_barcodes(barc2array(barcodes), title = '')
      plt.show()
    
    return barcodes
 
def calc_cross_barcodes(cloud_1, cloud_2, batch_size1 = 4000, batch_size2 = 200, cuda = 0, pdist_device = 'cpu', is_plot = False, dim = 1):
    dark_color_list = ['deepskyblue', 'maroon', 'g', 'darkorange']
    color_list = ['paleturquoise', 'lightcoral', 'lightgreen', 'lightsalmon']

    batch_size1 = min(batch_size1, cloud_1.shape[0])
    batch_size2 = min(batch_size2, cloud_2.shape[0])

    indexes_1 = np.random.choice(cloud_1.shape[0], batch_size1, replace=False)
    indexes_2 = np.random.choice(cloud_2.shape[0], batch_size2, replace=False)
    cl_1 = cloud_1[indexes_1]
    cl_2 = cloud_2[indexes_2]
    #barc_title = f'batch size {batch_size1,batch_size2}'
    barc_title = ''
    barc = count_cross_barcodes(cl_1, cl_2, dim, is_plot = is_plot, title = barc_title, cuda = cuda, pdist_device = pdist_device)

    return barc

def get_score(elem, h_idx, kind = ''):
    if elem.shape[0] >= h_idx + 1:

        barc = elem[h_idx]
        arrayForSort = np.subtract(barc[:,1], barc[:,0])

        bsorted = np.sort(arrayForSort)

        # number of barcodes
        if kind == 'nbarc':
            return bsorted.shape[0]

        # largest barcode
        if kind == 'largest':
            return bsorted[-1]

        # quantile
        if kind == 'quantile':
            idx = int(0.976 * len(bsorted))
            return bsorted[idx]

        # sum of length
        if kind == 'sum_length':
            return np.sum(bsorted)

        # sum of squared length
        if kind == 'sum_sq_length':
            return np.sum(bsorted**2)

        raise ValueError('Unknown kind of score')

    return 0

def mtopdiv(elem):
    return get_score(elem, 1, 'sum_length')

def h1sum(barc):
    if 1 in barc:
        return sum([x[1] - x[0] for x in barc[1]])
    else:
        return 0.0

def rtd1(cl1, cl2, pdist_device = 'cuda:0'):
    return (h1sum(calc_embed_dist(cl1, cl2, pdist_device = pdist_device)) +\
            h1sum(calc_embed_dist(cl2, cl1, pdist_device = pdist_device))) / 2

def rtd(cl1, cl2, pdist_device = 'cuda:0', trials = 10, batch = 500):

    assert cl1.shape[0] == cl2.shape[0]
    batch = min(batch, cl1.shape[0])

    rtd_avg = 0
    
    for i in range(trials):
        rnd_idx = list(range(cl1.shape[0]))
        np.random.shuffle(rnd_idx)
        rnd_idx = rnd_idx[:batch]
        rtd_avg += rtd1(cl1[rnd_idx], cl2[rnd_idx], pdist_device = pdist_device)
        
    return rtd_avg / trials

def rtd_dis(cl1, cl2, pdist_device = 'cuda:0', trials = 10, batch = 500):

    assert cl1.shape[0] == cl2.shape[0]
    batch = min(batch, cl1.shape[0])

    rtd_avg = 0
    
    for i in range(trials):

        rnd_idx = np.random.choice(cl1.shape[0], batch, replace=False) 
        cl1_subset = cl1[np.ix_(rnd_idx, rnd_idx)] 
        cl2_subset = cl2[np.ix_(rnd_idx, rnd_idx)] 
        rtd_avg += rtd1(cl1_subset, cl2_subset, pdist_device=pdist_device)
        
    return rtd_avg / trials