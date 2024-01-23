
import typing
import math
import numpy
import typing
import math
import numpy

class ImageNormalization(object):
    """
    Image normalization technique.
    Not channelwise
    """
    def _calculate_new_variance(self, variance: float) -> float:
       return variance
    def _calculate_new_mean(self, mean: float) -> float:
       return mean

    def normalize(self, input_image: numpy.ndarray):
        
        if not len(input_image): return input_image
        flat_data = input_image.flatten()
        
        mean = numpy.sum(flat_data) / flat_data.shape[0]
        var = numpy.sum(flat_data - mean) / flat_data.shape[0]
        
        new_variance = self._calculate_new_variance(variance=var)
        new_mean = self._calculate_new_mean(mean=mean)
 
        normalized_data = ((flat_data - new_mean) / math.sqrt(new_variance)).reshape(input_image.shape)
        scaled_data = self.gamma * normalized_data + self.beta 
        return numpy.clip(scaled_data, 0, 255.0)

