import numpy 


class SobelOperator(object):
    """
    Implementation of the Sobel Operator
    gradient-based edge detection filter
    """
    def __init__(self):
        self.gx = numpy.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).astype(numpy.float32)
        self.gy = numpy.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).astype(numpy.float32)
    
    def convert_to_grayscale(self, input_img: numpy.ndarray):
        gamma = 1.400
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = (r_const * input_img[:, :, 0] ** gamma) + \
        g_const * (input_img[:, :, 1] ** gamma) + b_const * (input_img[:, :, 2] ** gamma)
        return grayscale_image

    def round_direction_angle(self, angle: float):
        abs_angle = abs(angle)
        if abs_angle <= 22.5: return 0
        if abs_angle <= 67.5: return 45
        if abs_angle <= 112.5: return 90
        if abs_angle <= 157.5: return 135
        return 0
        
    def find_edges(self, input_img: numpy.ndarray):
        """
        Note:
            input img should be  
            grayscaled image with a single channel
            Additionally, for better processing and more
            meaningful results it is recommended to apply smoothing
            before the detector itself to remove noise in case of presence.

        Parameters:
        -----------
        gray_img - grayscale image
        """
        if len(input_img.shape) == 3:
            input_img = self.convert_to_grayscale(input_img=input_img)

        height, width = input_img.shape
        grad_img = numpy.zeros_like(input_img)
        grad_direction_map = numpy.zeros_like(input_img)

        for x in range(1, height - 2):
            for y in range(1, width - 2):
                
                gx = numpy.sum(numpy.multiply(self.gx, input_img[x:x+3, y:y+3]))
                gy = numpy.sum(numpy.multiply(self.gy, input_img[x:x+3, y:y+3]))

                grad_img[x+1, y+1] = numpy.sqrt((gx ** 2) + (gy ** 2))
                grad_direction_map[x+1, y+1] = self.round_direction_angle(numpy.arctan([gy / gx])[0])
                
        return grad_img, grad_direction_map