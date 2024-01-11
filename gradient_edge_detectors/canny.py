import numpy 
from cv_basics.gradient_edge_detectors.sobel import SobelOperator
from cv_basics.spatial_filters.gaussian_blur import gaussian_blur

class CannyDetector(object):
    """
    Implementation of the Canny Edge Detection Algorithm,
    based on Sobel operator
    """
    def __init__(self, 
        min_thresh: int, 
        max_thresh: int, 
        kernel_size: int, 
        sigmaX: float
    ):
        self.min_thresh = min_thresh 
        self.max_thresh = max_thresh 
        self.sobel = SobelOperator()
        self.sigmaX = sigmaX 
        self.kernel_size = kernel_size

    def non_max_suppression(self, 
        input_img: numpy.ndarray, 
        direction_img: numpy.ndarray, 
        grad_img: numpy.ndarray
    ):
        height, width = input_img.shape
        output_img = numpy.zeros_like(a=input_img)

        for x in range(1, height - 1):
            for y in range(1, width - 1):
                
                if direction_img[x, y] % 45 == 0:
                    if not (direction_img[x-1, y-1] <= direction_img[x, y] >= direction_img[x+1, y+1]):
                        output_img[x, y] = 0
                    else:
                        output_img[x, y] = grad_img[x, y]
    
                if direction_img[x, y] % 90 == 0:
                    if not (direction_img[x, y-1] <= direction_img[x, y] >= direction_img[x, y+1]):
                        output_img[x, y] = 0
                    else:
                        output_img[x, y] = grad_img[x, y]

                if direction_img[x, y] == 180:
                    if not (direction_img[x+1, y] <= direction_img[x, y] >= direction_img[x-1, y]):
                        output_img[x, y] = 0
                    else:
                        output_img[x, y] = grad_img[x, y]
        return output_img

    def compute(self, input_img: numpy.ndarray):

        if len(input_img.shape) == 3:
            raise ValueError(msg='required 1-channeled image')

        # applying smoothing to remove noise

        blurred_img = self.gaussian_blur(
            image=input_img, 
            kernel_size=self.kernel_size, 
            sigmaX=self.sigmaX
        )

        # applying Sobel Operator to find gradients and it's directions

        grad_img, direction_img = self.sobel.find_edges(gray_img=blurred_img)

        # applying non-maximum supression
        suppressed_img = self.non_max_suppression(input_img, direction_img, grad_img)

        # hysteresis thresholding 
        filtered_img = self.hysteresis_thresh(suppressed_img, direction_img, grad_img)
        return filtered_img 

