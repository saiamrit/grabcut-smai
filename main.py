import numpy as np
import cv2 as cv
from gmm import *
from grabcut import *
from ui import *
import warnings
warnings.filterwarnings("ignore")

gmm_components = 5
gamma = 30
neighbours = 8 
color_space = 'RGB'
n_iters = 10

def run(filename, gamma=50, gmm_components=7, neighbours=8, color_space='RGB'):
    """
    Main loop that implements GrabCut. 
    
    Input
    -----
    filename (str) : Path to image
    """
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
    mask = np.ones(img.shape[:2], dtype = np.uint8) * DRAW_PR_BG['val'] # mask is a binary array with : 0 - background pixels
                                                     #                               1 - foreground pixels 
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

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
        
        elif k == 13:
            
            # Press Enter key to start segmentation
            
            rect = FLAGS['RECT']
            mask = GrabCut(img2, mask, rect, gmm_components, gamma, neighbours, n_iters)
        
            EventObj.flags = FLAGS
            mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
            output = cv.bitwise_and(img2, img2, mask=mask2)


if __name__ == '__main__':
    # /home/saiamrit/Documents/CV assignments/assignment-3-saiamrit
    filename = './images/sheep.jpg'               # Path to image file
    run(filename, gamma, gmm_components, neighbours, color_space)
    cv.destroyAllWindows()