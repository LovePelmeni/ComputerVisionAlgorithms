import numpy

class HorizontalFlip(object):
    """
    Implementation of the Horizontal Flip
    augmentation function.
    """
    def flip(self, input_img: numpy.ndarray):
        output_img = numpy.empty(shape=input_img.shape)
        height, width = input_img.shape[:2]
        for x in range(0, width):
            for y in range(0, height):
                offset_x = int(numpy.clip(int(width - 1 - x), 0, width-1))
                output_img[y][offset_x][:] = input_img[y][x]
        return output_img.astype(numpy.uint8)

class VerticalFlip(object):
    """
    Implementation of the Vertical Flip augmentation.
    """
    def flip(self, input_img: numpy.ndarray):
        output_img = numpy.empty(shape=input_img.shape)
        height, width = input_img.shape[:2]
        for x in range(0, width):
            for y in range(0, height):
                offset_y = int(numpy.clip(int(height - 1 - y), 0, height-1))
                output_img[offset_y][x][:] = input_img[y][x][:]
        return output_img.astype(numpy.uint8)