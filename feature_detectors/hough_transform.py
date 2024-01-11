import cv2
import numpy 

class HoughTransform(object):

    def __init__(self, thresh1: int, thresh2: int):
        self.thresh1 = thresh1 
        self.thresh2 = thresh2

    def detect_edges(self):
        pass

    def apply(self, input_img: numpy.ndarray):
        pass


class HoughLines(HoughTransform):
    pass 


class HoughCircles(HoughTransform):
    pass


