import numpy
from loss import mse
import cv2

 
def psnr(input_img1, input_img2):

    if len(input_img1.shape) != 2:
        input_img1 = cv2.cvtColor(input_img1, cv2.COLOR_RGB2GRAY)

    if len(input_img2) != 2:
        input_img2 = cv2.cvtColor(input_img1, cv2.COLOR_RGB2GRAY)

    mse_error = mse(input_img1.flatten(), input_img2.flatten())
    return 10 * numpy.log10((255 ** 2) / mse_error)




