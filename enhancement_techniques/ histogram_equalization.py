import numpy


class HistogramEqualization(object):
    """
    Implementation of the Histogram Equalization
    Algorithm. Popular augmentation strategy 
    for maximizing constrast of the image.
    """
    def to_grayscale(self, input_img: numpy.ndarray):
        R = input_img[:, :, 0]
        G = input_img[:, :, 1]
        B = input_img[:, :, 2]
        return 0.299*R + 0.587*G + 0.114*B

    def apply(self, input_img: numpy.ndarray):

        if input_img.shape[-1] == 3:
            input_img = self.to_grayscale(input_img).astype('uint8')
  
        img_hist, _ = numpy.histogram(a=input_img, bins=256, range=[0, 256])
        cdf = numpy.cumsum(img_hist)
        cdf = numpy.ma.masked_equal(cdf, value=0)
        normalized_cdf = ((cdf - cdf.min()) / (cdf.max() - cdf.min())) * 255
        normalized_cdf = numpy.ma.filled(a=normalized_cdf, fill_value=0).astype('uint8')
        mapped_img = normalized_cdf[input_img]
        return mapped_img 