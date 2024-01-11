import numpy 

def gamma_correction(img: numpy.ndarray, gamma=1/2.2):
    """
    Implementation of the basic
    gamma correction method
    for regulating img intensity
    and ensuring correct visualization
    on the monitor, during image analysis

    Before embarking on any analysis,
    you need to correct your image 
    using this gamma correction method
    """
    corr_img = numpy.power(img, gamma).astype(numpy.uint8)
    return corr_img


