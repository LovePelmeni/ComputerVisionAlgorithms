import numpy 

class LaplacianOperator(object):
    """
    Implementation of the Laplacian Edge Detection
    Algorithm, based on second-order gradient approximation kernel
    """
    def __init__(self):
        self.operator = numpy.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])

    def convert_to_grayscale(self, input_img: numpy.ndarray):
        gamma = 1.400
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = (r_const * input_img[:, :, 0] ** gamma) + \
        g_const * (input_img[:, :, 1] ** gamma) + b_const * (input_img[:, :, 2] ** gamma)
        return grayscale_image
    
    def compute(self, input_img: numpy.ndarray):
        if len(input_img.shape) == 3:
            input_img = self.convert_to_grayscale(input_img)

        height, width = input_img.shape
        output_img = numpy.zeros_like(input_img.shape)

        for x in range(1, height - 1):
            for y in range(1, width - 1):
                output_img[x, y] = numpy.multiply(
                    self.operator, 
                    input_img[x-1:x+1, y-1:y+1]
                )
        return output_img