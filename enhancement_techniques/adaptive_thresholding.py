import numpy 
import typing
import sys 
sys.path.append("../")
from cv_basics.spatial_filters import gaussian_blur

class AdaptiveThreshold(object):

    def __init__(self, 
        tileSize: int, 
        adaptiveMethod: typing.Literal['gaussian', 'mean'],
        C=0.01 # constant for preventing division by zero
    ):
        self.tileSize = tileSize 
        self.adaptiveMethod = adaptiveMethod 
        self.C = C
        self.gaussian_function = gaussian_blur.GaussianBlur(
            sigmaX=2, 
            kernel_size=tileSize, 
            sigmaY=2
        )

    def pick_threshold(self, input_img: numpy.ndarray):

        if self.adaptiveMethod == 'mean':
            mean_thresh = numpy.mean(input_img) - self.C

        elif self.adaptiveMethod == 'gaussian':
            mean_thresh = numpy.sum(self.gaussian_function.blur(input_img))
            
        return mean_thresh

    def apply(self, input_img: numpy.ndarray):

        if self.tileSize % 2 == 0:
            raise ValueError('threshold value should be odd')

        if len(input_img.shape) != 2:
            raise TypeError("image should be grayscale")

        radius = self.tileSize // 2
        output_img = numpy.zeros_like(input_img)
        height, width = input_img.shape

        for x in range(radius, height-radius):
            for y in range(radius, width-radius):

                img_area = input_img[x-radius:x+radius+1, y-radius:y+radius+1]
            
                threshold = self.pick_threshold(img_area)

                output_img[x-radius:x+radius+1, y-radius:y+radius+1][img_area < threshold] = 0
                output_img[x-radius:x+radius+1, y-radius:y+radius+1][img_area >= threshold] = 1
                
        return output_img 

