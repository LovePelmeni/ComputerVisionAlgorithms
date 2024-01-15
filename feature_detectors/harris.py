import numpy 
class HarrisDetector(object):
    """
    Implementation of the Harris
    Corner Detection algorithm
    """
    def __init__(self, threshold: float, K: float = 0.05):
        self.dx_grad = [-1, 0, 1]
        self.dy_grad = [-1, 0, 1]
        self.threshold = threshold
        self.K = K

    def detect(self, input_img: numpy.ndarray):

        if len(input_img.shape) > 2:
            input_img = self.convert_to_grayscale(input_img)
            
        height, width = input_img.shape
        output_map = numpy.zeros(shape=input_img.shape)

        for x in range(1, height-1):
            for y in range(1, width-1):
                # computing gradients
                
                dx_grad = numpy.multiply(input_img[x-1:x+2, y], self.dx_grad)
                dy_grad = numpy.multiply(input_img[x, y-1:y+2], self.dy_grad)

                f00_m = numpy.sum(numpy.power(dx_grad, 2))
                f01_m = numpy.sum(numpy.power(dy_grad, 2))
                f10_m = numpy.sum(numpy.multiply(dx_grad, dy_grad))

                # computing corner response matrix
                
                harris_matrix = numpy.array([[f00_m, f10_m], [f10_m, f01_m]])
                
                minR = numpy.min(harris_matrix.flatten())
                maxR = numpy.max(harris_matrix.flatten())
                
                # normalizing corner response matrix
                harris_matrix = (harris_matrix - minR) / (maxR - minR)

                # computing corner magnitude
                
                corner_magnitude = (
                    numpy.linalg.det(harris_matrix) 
                    - self.K * (numpy.trace(harris_matrix) ** 2)
                )
        
                output_map[x, y] = int(corner_magnitude >= self.threshold) 
        return output_map