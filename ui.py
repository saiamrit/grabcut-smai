import sys
import os
import cv2 as cv
import warnings
warnings.filterwarnings("ignore")

class EventHandler:
    """
    Class for handling user input during segmentation iterations 
    """
    
    def __init__(self, flags, img, _mask, colors):
        
        self.FLAGS = flags
        self.ix = -1
        self.iy = -1
        self.img = img
        self.img2 = self.img.copy()
        self._mask = _mask
        self.COLORS = colors

    @property
    def image(self):
        return self.img
    
    @image.setter
    def image(self, img):
        self.img = img
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, _mask):
        self._mask = _mask
    
    @property
    def flags(self):
        return self.FLAGS 
    
    @flags.setter
    def flags(self, flags):
        self.FLAGS = flags
    
    def handler(self, event, x, y, flags, param):

        # Draw the rectangle first
        if event == cv.EVENT_RBUTTONDOWN:
            self.FLAGS['DRAW_RECT'] = True
            self.ix, self.iy = x,y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_RECT'] == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
                cv.rectangle(self._mask, (self.ix, self.iy), (x, y), self.FLAGS['value']['val'], -1)
                self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.FLAGS['rect_or_mask'] = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.FLAGS['DRAW_RECT'] = False
            self.FLAGS['rect_over'] = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
            cv.rectangle(self._mask, (self.ix, self.iy), (x, y), self.FLAGS['value']['val'], -1)
            self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.FLAGS['rect_or_mask'] = 0

        
        # Draw strokes for refinement 

        if event == cv.EVENT_LBUTTONDOWN:
            if self.FLAGS['rect_over'] == False:
                print('Draw the rectangle first.')
            else:
                self.FLAGS['DRAW_STROKE'] = True
                cv.circle(self.img, (x,y), 3, self.FLAGS['value']['color'], -1)
                cv.circle(self._mask, (x,y), 3, self.FLAGS['value']['val'], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_STROKE'] == True:
                cv.circle(self.img, (x, y), 3, self.FLAGS['value']['color'], -1)
                cv.circle(self._mask, (x, y), 3, self.FLAGS['value']['val'], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.FLAGS['DRAW_STROKE'] == True:
                self.FLAGS['DRAW_STROKE'] = False
                cv.circle(self.img, (x, y), 3, self.FLAGS['value']['color'], -1)
                cv.circle(self._mask, (x, y), 3, self.FLAGS['value']['val'], -1)