import numpy as np
import cv2 as cv
from gmm import *
from grabcut import *
from ui import *
import warnings
import argparse
warnings.filterwarnings("ignore")

gmm_components = 5
gamma = 30
neighbours = 8 
color_space = 'RGB'
n_iters = 10

def run(filename, gamma=50, gmm_components=7, neighbours=8, n_iters=5, color_space='RGB'):
    """
    Main loop that implements GrabCut. 
    
    Input
    -----
    filename (str) : Path to image
    """
    
    COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

    DRAW_PR_BG = {'color' : COLORS['BLACK'], 'val' : 2}
    DRAW_PR_FG = {'color' : COLORS['WHITE'], 'val' : 3}
    DRAW_BG = {'color' : COLORS['BLACK'], 'val' : 0}
    DRAW_FG = {'color' : COLORS['WHITE'], 'val' : 1}

    FLAGS = {
        'RECT' : (0, 0, 1, 1),
        'DRAW_STROKE': False,         # flag for drawing strokes
        'DRAW_RECT' : False,          # flag for drawing rectangle
        'rect_over' : False,          # flag to check if rectangle is  drawn
        'rect_or_mask' : -1,          # flag for selecting rectangle or stroke mode
        'value' : DRAW_PR_FG,            # drawing strokes initialized to mark foreground
    }

    img = load_image(filename, color_space, scale=0.8)
    img2 = img.copy()                                
    mask = np.ones(img.shape[:2], dtype = np.uint8) * DRAW_PR_BG['val'] 
                                        # mask is a binary array with : 
                                                    # 0 - background pixels
                                                    # 1 - foreground pixels 
    output = np.zeros(img.shape, np.uint8)          # output image to be shown

    # Input and segmentation windows
    cv.namedWindow('Input Image')
    cv.namedWindow('Segmented image')
    
    
    EventObj = EventHandler(FLAGS, img, mask, COLORS)
    cv.setMouseCallback('Input Image', EventObj.handler)
    cv.moveWindow('Input Image', img.shape[1] + 10, 90)
    
    while(1):
        
        img = EventObj.image
        mask = EventObj.mask
        FLAGS = EventObj.flags
        cv.imshow('Segmented image', output)
        cv.imshow('Input Image', img)
        
        k = cv.waitKey(1)

        # key bindings
        if k == 27:
            # esc to exit
            break
        
        elif k == ord('0'): 
            # Strokes for background
            FLAGS['value'] = DRAW_BG
        
        elif k == ord('1'):
            # FG drawing
            FLAGS['value'] = DRAW_FG
        
        elif k == ord('r'):
            # reset everything
            FLAGS['RECT'] = (0, 0, 1, 1)
            FLAGS['DRAW_STROKE'] = False
            FLAGS['DRAW_RECT'] = False
            FLAGS['rect_or_mask'] = -1
            FLAGS['rect_over'] = False
            FLAGS['value'] = DRAW_PR_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype = np.uint8) 
            EventObj.image = img
            EventObj.mask = mask
            output = np.zeros(img.shape, np.uint8)
        
        elif k == 13:
            # Press Enter to initiate segmentation
            
            gc = GrabCut(img=img2, mask=mask, n_iters=n_iters, gamma=gamma, gmm_components=gmm_components, neighbours=neighbours)
            mask = gc.mask.copy()
        
            EventObj.flags = FLAGS
            mask2 = np.where(np.logical_or((mask == 3), (mask==1)), 255, 0).astype('uint8')
            output = cv.bitwise_and(img2, img2, mask = mask2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Grabcut Segmentation parameters Parser. Pass the parameters following instructions given\
    below to run the demo experiment.")
    # Input and output paths
    parser.add_argument('--input_path', default="./images/sheep.jpg", type=str, help='Path to image for processing')
    # parser.add_argument('--output', default="./output/", type=str, help="Path to folder for storing output")
    
    # Segmentation parameters
    parser.add_argument('--n_iters', default=5, type=int, help="No of iterations to run")
    parser.add_argument('--gamma', default=30, type=int, help="Value for Gamma parameter")
    parser.add_argument('--gmm_components', default=5, type=int, help="Number of GMM components for each GMM")
    parser.add_argument('--neighbours', default=8, type=int, help="8 or 4 connectivity to be considered")
    parser.add_argument('--color_space', default='RGB', type=str, help="Parameter value for beta")


    args = parser.parse_args()

    gmm_components = args.gmm_components
    gamma = args.gamma
    neighbours = args.neighbours
    color_space = args.color_space
    n_iters = args.n_iters


    filename = args.input_path               # Path to image file
    run(filename, gamma, gmm_components, neighbours, n_iters, color_space)
    cv.destroyAllWindows()