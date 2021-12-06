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

def load_image(filename, color_space = 'RGB', scale=0.75):
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

class GrabCut:
    def __init__(self, img, mask, n_iters, gamma=50, gmm_components=5, neighbours=8, rect=None):
        # Save the image and it's size first
        self.img = img.copy()
        self.rows, self.cols, _ = self.img.shape
        self.mask = mask.copy()
        
        # Set the mask with the probable foreground
        if rect is not None:
            self.mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = DRAW_PR_FG['val']
        
        # plt.imshow(self.mask)
        
        # Update pixel labels from the bounding box
        self.classify_pixels()
        
        # Set the parameters
        self.gamma = gamma
        self.gmm_components = gmm_components
        self.n_iters = n_iters
        self.neighbours = neighbours
        
        # beta value for the smoothness term in the Gibbs energy function
        self.beta = 0
        
        # Placeholders for shifted image values for pairwise computations
        self.img_left = np.empty((self.rows, self.cols - 1))
        self.img_up = np.empty((self.rows - 1, self.cols))
        
        # Add more placeholders if we want 8 connectivity
        if self.neighbours == 8:
            self.img_upleft = np.empty((self.rows - 1, self.cols - 1))
            self.img_upright = np.empty((self.rows - 1, self.cols - 1))
            
        
        # Placeholders for GMM models
        self.gmm_fg = None
        self.gmm_bg = None
        self.comp_idx = np.empty((self.rows, self.cols), dtype=np.uint32)
        
        self.graph = None
        self.graph_capacity = None
        self.graph_source  = self.rows * self.cols
        self.graph_sink = self.graph_source + 1
        
        # Now, we initialise the Gibbs functions and GMMs
        self.compute_smoothness()
        self.initialise_GMMs()
        
        # Start the GrabCut method
        self.run()
        
        
    def classify_pixels(self):#     print('bgd count: %d, fgd count: %d, uncertain count: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))

        # Allocate the indices with foregorund and background pixels
        self.bg_idx = np.where(np.logical_or(self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))
        self.fg_idx = np.where(np.logical_or(self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))
        # print(self.bg_idx[0].shape, self.fg_idx[0].shape)
    
    def compute_smoothness(self):
        """
        Compute the pairwise smoothness term from the Gibbs energy function
        """
        # First compute all the differences
        diff_left = self.img[:, 1:] - self.img[:, :-1]
        diff_up = self.img[1:, :] - self.img[:-1, :]
        
        # Compute the sum of difference terms to get beta value
        beta_sum = np.sum(np.square(diff_left)) + np.sum(np.square(diff_up))
        # We need to divide the sum above to get the expectation value
        n_avg = (2 * self.rows * self.cols) - self.cols - self.rows
        
        # Compute addditional difference if 8-neighbours are considered
        if self.neighbours == 8:
            diff_upleft = self.img[1:, 1:] - self.img[:-1, :-1]
            diff_upright = self.img[1:, :-1] - self.img[:-1, 1:]
            
            # Update the beta related values as well
            beta_sum += np.sum(np.square(diff_upleft)) + np.sum(np.square(diff_upright))
            n_avg += (2 * self.rows * self.cols) - (2 * self.cols) - (2 * self.rows) + 2
        
        # Finally, compute the beta value
        self.beta = n_avg / (2 * beta_sum)
        
        # Now, we need to compute the smoothness term V
        self.V_left = self.gamma * np.exp(-self.beta * np.sum(np.square(diff_left), axis=2))
        self.V_up = self.gamma * np.exp(-self.beta * np.sum(np.square(diff_up), axis=2))
        
        # Again, for the 8-neighbour case
        if self.neighbours == 8:
            self.V_upleft = self.gamma * np.exp(-self.beta * np.sum(np.square(diff_upleft), axis=2))
            self.V_upright = self.gamma * np.exp(-self.beta * np.sum(np.square(diff_upright), axis=2))
    
    def initialise_GMMs(self):
        # Initialise the GMM objects here
        self.gmm_bg = GMM(self.img[self.bg_idx], self.gmm_components)
        self.gmm_fg = GMM(self.img[self.fg_idx], self.gmm_components)
    
    def run(self):
        for iter_ in range(self.n_iters):
            # print("Current iteration: {}".format(iter_))
            self.assign_gmm_components()
            # print("Assigned GMM components")
            self.initialise_GMMs()
            # print("Initialized GMMs for this iteration")
            self.create_graph()
            # print("Completed the creation of graphs")
            self.estimate_segmentation()
            # print("Estimated the segmentation")
    
    def assign_gmm_components(self):
        #Step 1 in Figure 3: Assign GMM components to pixels
        self.comp_idx[self.bg_idx] = self.gmm_bg.get_components(self.img[self.bg_idx])
        self.comp_idx[self.fg_idx] = self.gmm_fg.get_components(self.img[self.fg_idx])
    
    def create_graph(self):
        # Get the indices of foreground, background and soft labels
        bg_idx = np.where(self.mask.reshape(-1) == DRAW_BG['val'])
        fg_idx = np.where(self.mask.reshape(-1) == DRAW_FG['val'])
        pr_idx = np.where(np.logical_or(self.mask.reshape(-1) == DRAW_PR_BG['val'], self.mask.reshape(-1) == DRAW_PR_FG['val']))
        pr_idx_ = np.where(np.logical_or(self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        
        # Prepare to create the graph
        edges = []
        self.graph_capacity = []
        
        # Create t-links (connect nodes to terminal nodes)
        # Prob values to source
        edges.extend(
            list(zip([self.graph_source for ix in range(pr_idx[0].size)], pr_idx[0]))
        )
        
        start_time = datetime.datetime.now()
        start = datetime.datetime.now()
        _D = self.gmm_bg.compute_D(self.img[pr_idx_])
        self.graph_capacity.extend(_D.tolist())
        # print("Background weight mean: {} | time taken: {}".format(_D.mean(), datetime.datetime.now() - start))
        
        
        # prob values to sink
        start = datetime.datetime.now()
        edges.extend(
            list(zip([self.graph_sink for ix in range(pr_idx[0].size)], pr_idx[0]))
        )
        # print("time taken for edge extend: {}".format(datetime.datetime.now() - start))
        
        
        start = datetime.datetime.now()
        _D = self.gmm_fg.compute_D(self.img[pr_idx_])
        self.graph_capacity.extend(_D.tolist())
        # print("Foreground weight mean: {} | time taken: {}".format(_D.mean(), datetime.datetime.now() - start))
        # print("Time for whole chunk: {}".format(datetime.datetime.now() - start_time))
        # Background to source
        # print("Edges before FG abd BG: {}".format(len(edges)))
        edges.extend(
            list(zip([self.graph_source for ix in range(bg_idx[0].size)], bg_idx[0]))
        )
        self.graph_capacity.extend([0] * bg_idx[0].size)
        
        # background to sink
        edges.extend(
            list(zip([self.graph_sink for ix in range(bg_idx[0].size)], bg_idx[0]))
        )
        self.graph_capacity.extend([99 * self.gamma] * bg_idx[0].size)
        
        # Foreground to source
        edges.extend(
            list(zip([self.graph_source for ix in range(fg_idx[0].size)], fg_idx[0]))
        )
        self.graph_capacity.extend([99 * self.gamma] * fg_idx[0].size)
        
        # Foreground to sink
        edges.extend(
            list(zip([self.graph_sink for ix in range(fg_idx[0].size)], fg_idx[0]))
        )
        self.graph_capacity.extend([0] * fg_idx[0].size)
        # print("Edges after FG abd BG: {}".format(len(edges)))
        
        
        # Now we create n-links (connect nodes to other nodes (non-terminal))
        img_indexes = np.arange(self.rows * self.cols, dtype=np.uint32).reshape(self.rows, self.cols)
        
        # get shifted indices and connect the points (left)
        mask1 = img_indexes[:, 1:].reshape(-1)
        mask2 = img_indexes[:, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.graph_capacity.extend(self.V_left.reshape(-1).tolist())
        
        # get shifted indices and connect the points (up)
        mask1 = img_indexes[1:, :].reshape(-1)
        mask2 = img_indexes[:-1, :].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.graph_capacity.extend(self.V_up.reshape(-1).tolist())
        
        # For 8-connectivity
        if self.neighbours == 8:
            # get shifted indices and connect the points (up-left)
            mask1 = img_indexes[1:, 1:].reshape(-1)
            mask2 = img_indexes[:-1, :-1].reshape(-1)
            edges.extend(list(zip(mask1, mask2)))
            self.graph_capacity.extend(self.V_upleft.reshape(-1).tolist())
            
            # get shifted indices and connect the points (up-right)
            mask1 = img_indexes[1:, :-1].reshape(-1)
            mask2 = img_indexes[:-1, 1:].reshape(-1)
            edges.extend(list(zip(mask1, mask2)))
            self.graph_capacity.extend(self.V_upright.reshape(-1).tolist())
        
        # Construct the graph and add the edges
        self.graph = ig.Graph(self.cols * self.rows + 2)
        self.graph.add_edges(edges)
    
    def estimate_segmentation(self):
        # Apply the mincut algorithm first
        start = datetime.datetime.now()
        mincut = self.graph.st_mincut(self.graph_source, self.graph_sink, self.graph_capacity)
        
        # print("Mincut time: {}".format(datetime.datetime.now() - start))
        # Compute the probability indices
        pr_idx = np.where(np.logical_or(self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        img_idx = np.arange(self.rows * self.cols, dtype=np.uint32).reshape((self.rows, self.cols))
        
        # Update mask with foregrund and background values and update indices
        self.mask[pr_idx] = np.where(np.isin(img_idx[pr_idx], mincut.partition[0]), DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        
        # print(np.unique(self.mask.flatten(), return_counts=True))
        
        self.classify_pixels()