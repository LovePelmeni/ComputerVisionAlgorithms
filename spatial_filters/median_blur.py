import numpy
class MedianBlur(object):

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size 
        self.kernel = numpy.ones(shape=(kernel_size, kernel_size))

    def blur(self, input_img: numpy.ndarray):

        height, width = input_img.shape
        radius = self.kernel_size // 2
        output_img = numpy.zeros(shape=input_img.shape)

        for x in range(height):
            for y in range(width):
                conv = numpy.multiply(
                    self.kernel,
                    output_img[
                        x-radius:x+radius+1, 
                        y-radius:y+radius+1
                    ]
                )
                input_img[x, y] = numpy.median(conv)
        return output_img