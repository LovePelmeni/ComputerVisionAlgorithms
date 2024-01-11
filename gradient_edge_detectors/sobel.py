import numpy 


class SobelOperator(object):
    """
    Implementation of the Sobel Operator
    gradient-based edge detection filter
    """
    def __init__(self):
        self.gx = numpy.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).astype(numpy.float32)
        self.gy = numpy.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).atype(numpy.float32)
    
    def convert_to_grayscale(self, input_img: numpy.ndarray):
        gamma = 1.400
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = (r_const * input_img[:, :, 0] ** gamma) + \
        g_const * (input_img[:, :, 1] ** gamma) + b_const * (input_img[:, :, 2] ** gamma)
        return grayscale_image

    def get_directions_map(self, input_img: numpy.ndarray):
        pass

    def find_edges(self, gray_img: numpy.ndarray):
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
        if gray_img.shape[2] == 3:
            gray_img = self.convert_to_grayscale(input_img=gray_img)

        height, width = gray_img.shape
        output = numpy.zeros_like(gray_img)

        for x in range(1, height - 2):
            for y in range(1, width - 2):
                gx = numpy.sum(numpy.multiply(self.gx, gray_img[x:x+3, y:y+3]))
                gy = numpy.sum(numpy.multiply(self.gy, gray_img[x:x+3, y:y+3]))

                output[x+1, y+1] = numpy.sqrt((gx ** 2) + (gy ** 2))
        return output 

