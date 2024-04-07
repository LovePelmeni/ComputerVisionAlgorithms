import numpy
import math
import cv2

class RotateAngleAugmentation(object):
    """
    Basic image rotation affine transformation.
    That rotates image by the parameter of 'angle'.
    """
    def __init__(self, angle: float, prob: float = 1):
        if angle < 0:
            self.angle = (180 - angle) * numpy.pi / 180.0
        else:
            self.angle = angle * numpy.pi / 180.0
        self.prob = prob
        self.rot_matrix = numpy.array(
            [
                [numpy.cos(angle), numpy.sin(angle)],
                [-numpy.sin(angle), numpy.cos(angle)]
            ]
        ).astype(numpy.float16)

    def affine_pixel(self, x, y):
        xy_vector = numpy.array([x, y], dtype=numpy.float32)
        new_xy_vector = numpy.dot(self.rot_matrix, xy_vector)
        return new_xy_vector[0], new_xy_vector[1]

    def linear_interpolation(self, coordinate: list):
        """
        Used for predicting more precise location
        of a pixel after applying affine transformation.
        """

    def apply(self, input_img: numpy.ndarray):

        height, width, num_channels = input_img.shape
        rotated_len = int(math.sqrt(math.pow(height, 2) + math.pow(width, 2)))

        rotated_img = numpy.empty(
            shape=(
                rotated_len, 
                rotated_len, 
                num_channels
            ), dtype=numpy.uint8
        )
        rot_w, rot_h = rotated_img.shape[:2]
        offset = int((rot_w+1)/2) # suppose we rotate, relative to center coordinate

        for x in range(rot_w):
            for y in range(rot_h):

                rot_x = (x - offset) * math.cos(self.angle) + (y - offset) * math.sin(self.angle)
                rot_y = -(x - offset) * math.sin(self.angle) + (y - offset) * math.cos(self.angle)

                # getting nearest neighbor index, however can
                # be solved using linear interpolation, as I've looked up
                # from response on stack overflow
                rot_x = self.linear_interpolation(rot_x + offset)
                rot_y = self.linear_interpolation(rot_y + offset)

                if (rot_x >= 0 and rot_y >= 0 and rot_x < width and rot_y < height):
                    rotated_img[y][x][:] = input_img[rot_y][rot_x][:]
        return rotated_img


if __name__ == "__main__":
    rotator = RotateAngleAugmentation(angle=45)
    image = cv2.imread("cv_basics/test_images/test_image.jpeg", cv2.IMREAD_UNCHANGED)
    rot_image = rotator.apply(input_img=image)
    cv2.imwrite(filename="../file.jpeg", img=rot_image)
