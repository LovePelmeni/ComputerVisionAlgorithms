import numpy

class HorizontalShearing(object):
    """
    Implementation of the horizontal
    shearing operation of the image.

    Parameters:
    ----------
        - shearing_angle - angle to use for shearing across horizontal axis
    """
    def __init__(self, shearing_angle: float):
        rad_shearing_angle = int(shearing_angle * numpy.pi / 180.0)
        shearing_factor: float = numpy.cos(rad_shearing_angle) / numpy.sin(rad_shearing_angle)
        self.shearing_matrix = self._generate_shear_matrix(shearing_factor)
    
    def _generate_shear_matrix(self, factor: float) -> numpy.ndarray:
        return numpy.array(
            [
                [1, factor],
                [0, 1]
            ]
        )

    def transform(self, input_img: numpy.ndarray):
        output_img = numpy.zeros(shape=input_img.shape)
        height, width = input_img.shape[:2]
        for x in range(width):
            for y in range(height):
                shear_x, shear_y = numpy.dot(self.shearing_matrix, numpy.array([x, y]))
                shear_x = int(numpy.clip(shear_x, 0, width-1))
                shear_y = int(numpy.clip(shear_y, 0, height-1))
                output_img[shear_y][shear_x][:] = input_img[y][x][:]
        return output_img.astype(numpy.uint8)

class VerticalShearing(object):
    """
    Implementation of the vertical
    shearing operation of the image.
     
    Parameters:
    -----------
        shearing_angle - angle to use for shearing across vertical axis
    """
    def __init__(self, shearing_angle: int):
        rad_shearing_angle = int(shearing_angle * numpy.pi / 180.0)
        shearing_factor: float = round(numpy.cos(rad_shearing_angle) / numpy.sin(rad_shearing_angle), 5)
        self.shearing_matrix = self._generate_shear_matrix(shearing_factor)
    
    def _generate_shear_matrix(self, factor: float) -> numpy.ndarray:
        return numpy.array(
            [
                [1, 0],
                [factor, 1]
            ]
        )

    def transform(self, input_img: numpy.ndarray):
        output_img = numpy.zeros(shape=input_img.shape)
        height, width = input_img.shape[:2]
        for x in range(width):
            for y in range(height):
                shear_x, shear_y = numpy.dot(self.shearing_matrix, numpy.array([x, y]))
                shear_x = int(numpy.clip(shear_x, 0, width-1))
                shear_y = int(numpy.clip(shear_y, 0, height-1))
                output_img[shear_y][shear_x][:] = input_img[y][x][:]
        return output_img.astype(numpy.uint8)