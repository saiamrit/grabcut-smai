import numpy as np
import math
import cv2 as cv
import igraph as ig
from gmm import *
import warnings
warnings.filterwarnings("ignore")

COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

gmm_components = 5
gamma = 30
neighbours = 8 
color_space = 'RGB'
n_iters = 10

DRAW_PR_BG = {'color' : COLORS['BLACK'], 'val' : 2}
DRAW_PR_FG = {'color' : COLORS['WHITE'], 'val' : 3}
DRAW_BG = {'color' : COLORS['BLACK'], 'val' : 0}
DRAW_FG = {'color' : COLORS['WHITE'], 'val' : 1}

def load_image(filename, color_space='RGB', scale=0.5):
    im = cv.imread(filename)
    if color_space == "RGB":
        pass
#         im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    elif color_space == "HSV":
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    elif color_space == "LAB":
        im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    if not scale == 1.0:
        im = cv.resize(im, (int(im.shape[1]*scale), int(im.shape[0]*scale)))
    return im


def construct_gc_graph(img,mask,gc_source,gc_sink,fgd_gmm,bgd_gmm,gamma,rows,cols,left_V, up_V, neighbours, upleft_V=None,upright_V=None):
    bgd_indexes = np.where(mask.reshape(-1) == DRAW_BG['val'])
    fgd_indexes = np.where(mask.reshape(-1) == DRAW_FG['val'])
    pr_indexes = np.where(np.logical_or(mask.reshape(-1) == DRAW_PR_BG['val'],mask.reshape(-1) == DRAW_PR_FG['val']))
#     print('bgd count: %d, fgd count: %d, uncertain count: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))
    edges = []
    gc_graph_capacity = []
    
    edges.extend(list(zip([gc_source] * pr_indexes[0].size, pr_indexes[0])))
    _D = -np.log(bgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    gc_graph_capacity.extend(_D.tolist())
    
    edges.extend(list(zip([gc_sink] * pr_indexes[0].size, pr_indexes[0])))
    _D = -np.log(fgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    gc_graph_capacity.extend(_D.tolist())
    
    edges.extend(list(zip([gc_source] * bgd_indexes[0].size, bgd_indexes[0])))
    gc_graph_capacity.extend([0] * bgd_indexes[0].size)
    edges.extend(list(zip([gc_sink] * bgd_indexes[0].size, bgd_indexes[0])))
    gc_graph_capacity.extend([9 * gamma] * bgd_indexes[0].size)
    edges.extend(list(zip([gc_source] * fgd_indexes[0].size, fgd_indexes[0])))
    gc_graph_capacity.extend([9 * gamma] * fgd_indexes[0].size)
    edges.extend(list(zip([gc_sink] * fgd_indexes[0].size, fgd_indexes[0])))
    gc_graph_capacity.extend([0] * fgd_indexes[0].size)

    img_indexes = np.arange(rows*cols,dtype=np.uint32).reshape(rows,cols)
    temp1 = img_indexes[:, 1:]
    temp2 = img_indexes[:, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(left_V.reshape(-1).tolist())
    
    temp1 = img_indexes[1:, 1:]
    temp2 = img_indexes[:-1, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(up_V.reshape(-1).tolist())
    
    if neighbours == 8:
        temp1 = img_indexes[1:, :]
        temp2 = img_indexes[:-1, :]
        mask1 = temp1.reshape(-1)
        mask2 = temp2.reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        gc_graph_capacity.extend(upleft_V.reshape(-1).tolist())
        
        temp1 = img_indexes[1:, :-1]
        temp2 = img_indexes[:-1, 1:]
        mask1 = temp1.reshape(-1)
        mask2 = temp2.reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        gc_graph_capacity.extend(upright_V.reshape(-1).tolist())
        
    gc_graph = ig.Graph(cols * rows + 2)
    gc_graph.add_edges(edges)
    return gc_graph,gc_source,gc_sink,gc_graph_capacity

def estimate_segmentation(mask,gc_graph,gc_source,gc_sink,gc_graph_capacity,rows,cols):
    mincut = gc_graph.st_mincut(gc_source,gc_sink, gc_graph_capacity)
#     print('foreground pixels: %d, background pixels: %d' % (len(mincut.partition[0]), len(mincut.partition[1])))
    pr_indexes = np.where(np.logical_or(mask == DRAW_PR_BG['val'], mask == DRAW_PR_FG['val']))
    img_indexes = np.arange(rows * cols,dtype=np.uint32).reshape(rows, cols)
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),DRAW_PR_FG['val'], DRAW_PR_BG['val'])
    bgd_indexes = np.where(np.logical_or(mask == DRAW_BG['val'],mask == DRAW_PR_BG['val']))
    fgd_indexes = np.where(np.logical_or(mask == DRAW_FG['val'],mask == DRAW_PR_FG['val']))
#     print('probble background count: %d, probable foreground count: %d' % (bgd_indexes[0].size,fgd_indexes[0].size))
    return pr_indexes,img_indexes,mask,bgd_indexes,fgd_indexes

def classify_pixels(mask):
    bgd_indexes = np.where(np.logical_or(mask == DRAW_BG['val'], mask == DRAW_PR_BG['val']))
    fgd_indexes = np.where(np.logical_or(mask == DRAW_FG['val'], mask == DRAW_PR_FG['val']))
    return fgd_indexes, bgd_indexes

def compute_smoothness(img, rows, cols, neighbours):
    left_diff = img[:, 1:] - img[:, :-1]
    up_diff = img[1:, :] - img[:-1, :]
    sq_left_diff = np.square(left_diff)
    sq_up_diff = np.square(up_diff)
    beta_sum = (np.sum(sq_left_diff) + np.sum(sq_up_diff))
    avg = (2 * rows * cols) - cols - rows

    if neighbours == 8:
        upleft_diff = img[1:, 1:] - img[:-1, :-1]
        upright_diff = img[1:, :-1] - img[:-1, 1:]
        sq_upleft_diff = np.square(upleft_diff)
        sq_upright_diff = np.square(upright_diff)
        beta_sum += np.sum(sq_upleft_diff) + np.sum(sq_upright_diff)
        avg += (2 * rows * cols) - (2 * cols) - (2 * rows) + 2

    
    beta = avg / (2 * beta_sum)
#     print('Beta:',beta)
    left_V = gamma * np.exp(-beta * np.sum(np.square(left_diff), axis=2))
    up_V = gamma * np.exp(-beta * np.sum(np.square(up_diff), axis=2))
    
    if neighbours == 8:
        upleft_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(upleft_diff), axis=2))
        upright_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(upright_diff), axis=2))
        return gamma, left_V, up_V, upleft_V, upright_V
    else:
        return gamma, left_V, up_V, None, None

def initialize_gmm(img, bgd_indexes, fgd_indexes, gmm_components):
    bgd_gmm = GaussianMixture(img[bgd_indexes], gmm_components)
    fgd_gmm = GaussianMixture(img[fgd_indexes], gmm_components)
    
    return fgd_gmm, bgd_gmm

def GrabCut(img, mask, rect, gmm_components, gamma, neighbours, n_iters):
    img = np.asarray(img, dtype=np.float64)
    rows,cols, _ = img.shape
    if rect is not None:
        mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = DRAW_PR_FG['val']

    fgd_indexes, bgd_indexes = classify_pixels(mask)
    
    gmm_components = gmm_components
    gamma = gamma
    beta = 0
    neighbours = neighbours
    
    left_V = np.empty((rows,cols - 1))
    up_V = np.empty((rows - 1,cols))
    
    if neighbours == 8:
        upleft_V = np.empty((rows - 1,cols - 1))
        upright_V = np.empty((rows - 1,cols - 1))
        
    bgd_gmm = None
    fgd_gmm = None
    
    comp_idxs = np.empty((rows,cols), dtype=np.uint32)
    
    gc_graph = None
    gc_graph_capacity = None
    gc_source = cols*rows
    gc_sink = gc_source + 1

    gamma, left_V, up_V, upleft_V, upright_V = compute_smoothness(img, rows, cols, neighbours)
    fwd_gmm, bgd_gmm = initialize_gmm(img, bgd_indexes, fgd_indexes, gmm_components)
    
    n_iters = n_iters
    for iters in range(n_iters):
        fgd_gmm, bgd_gmm = initialize_gmm(img, bgd_indexes, fgd_indexes, gmm_components)
        
        if neighbours == 8:
            gc_graph,gc_source,gc_sink,gc_graph_capacity = construct_gc_graph(img,mask,gc_source,gc_sink,
                                                                              fgd_gmm,bgd_gmm,gamma,rows,
                                                                              cols,left_V, up_V, neighbours, 
                                                                              upleft_V, upright_V)
        else:
            gc_graph,gc_source,gc_sink,gc_graph_capacity = construct_gc_graph(img,mask,gc_source,gc_sink,
                                                                              fgd_gmm,bgd_gmm,gamma,rows,
                                                                              cols,left_V, up_V, neighbours,
                                                                              upleft_V=None, upright_V=None)

        pr_indexes,img_indexes,mask,bgd_indexes,fgd_indexes = estimate_segmentation(mask,gc_graph,gc_source,
                                                                                    gc_sink,gc_graph_capacity,
                                                                                    rows,cols)
    return mask