import numpy 
from cv_basics.gradient_edge_detectors.sobel import SobelOperator
from cv_basics.spatial_filters.gaussian_blur import GaussianBlur


class CannyDetector(object):
    """
    Implementation of the Canny Edge Detection Algorithm,
    based on Sobel operator
    """
    def __init__(self, 
        low_threshold: int, 
        high_threshold: int, 
        sigmaX: float,
        sigmaY: float,
        gaussian_kernel_size: int
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sobel = SobelOperator()
        self.gaussian_blur = GaussianBlur(
            sigmaX=sigmaX, 
            sigmaY=sigmaY, 
            kernel_size=gaussian_kernel_size
        )

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


    def hysteresis_thresholding(self, input_img: numpy.ndarray):

        output = numpy.zeros(shape=input_img.shape)
        height, width = input_img.shape 

        # Iterate over each pixel in the image
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # If the pixel intensity is greater than the high threshold, mark it as an edge
                if input_img[i][j] > self.high_threshold:
                    output[i][j] = 1
                # If the pixel intensity is less than the low threshold, ignore it
                elif input_img[i][j] < self.low_threshold:
                    continue
                # If the pixel intensity is between the low and high thresholds, check its neighbors
                else:
                    # Get the 8-connected neighborhood of the pixel
                    neighbors = [(i+di, j+dj) for di in [-1, 0, 1] for dj in [-1, 0, 1] if (di != 0 or dj != 0)]
                    
                    # Check if any neighbor is an edge
                    for ni, nj in neighbors:
                        if (self.low_threshold <= input_img[ni][nj]):
                            output[i][j] = 1
                            break
        return output

    def compute(self, input_img: numpy.ndarray):

        if len(input_img.shape) == 3:
            raise ValueError(msg='required 1-channeled image')

        # applying smoothing to remove noise

        blurred_img = self.gaussian_blur.blur(input_img=input_img)
        
        # applying Sobel Operator to find gradients and it's directions

        grad_img, direction_img = self.sobel.find_edges(input_img=blurred_img)

        # applying non-maximum supression
        suppressed_img = self.non_max_suppression(input_img, direction_img, grad_img)
        
        # hysteresis thresholding 
        output_img = self.hysteresis_thresholding(suppressed_img)
        return output_img