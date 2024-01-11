import numpy 
class GaussianBlur(object):

    def __init__(self, sigmaX: float, sigmaY: float, kernel_size: int):

        if kernel_size % 2 == 0:
            raise ValueError(msg='kernel size should be odd, not even')

        self.sigmaX = sigmaX 
        self.sigmaY = sigmaY 
        self.kernel_size = kernel_size
        self.kernel = self._generate_gaussian_kernel()

    def _split_channels(self, input_img: numpy.ndarray):
        total_channels = len(input_img[0, 0, :])
        channels = []
        for ch in range(total_channels):
            channels.append(input_img[:, :, ch])
        return channels
    
    def _generate_gaussian_kernel(self):
        """
        Function generates gaussian kernel
        """
        output_kernel = numpy.zeros(shape=(self.kernel_size, self.kernel_size))
        radius = (self.kernel_size // 2)

        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                output_kernel[x, y] = self.gaussian_eq(
                    posX=x, 
                    posY=y, 
                    X0=radius, 
                    Y0=radius
                )
        return output_kernel

    def get_normalized_output(self, matrix_kernel: numpy.ndarray):
        """
        Function normalizes the output of the convolution of gaussian kernel 
        and region of the image. 
        It's done, because Gaussian is weighted smoothing filter
        """
        return numpy.sum(matrix_kernel) / (self.kernel.flatten().sum())

    def gaussian_eq(self, posX: int, posY: int, X0: int, Y0: int):
        """
        Gaussian function for 2D images
        """
        x_part = numpy.power((posX - X0), 2) / (2 * numpy.power(self.sigmaX, 2))
        y_part = numpy.power((posY - Y0), 2) / (2 * numpy.power(self.sigmaY, 2))
        exp_power = - (y_part + x_part)
        return (1 / 2 * numpy.pi * numpy.power(self.sigmaX, 2)) * numpy.exp(exp_power)

    def blur_channel(self, input_channel: numpy.ndarray):

        height, width = input_channel.shape 
        output_img = numpy.zeros_like(a=input_channel)
        radius = self.kernel_size // 2

        for x in range(radius, height - radius):
            for y in range(radius, width - radius):
                convolution = numpy.multiply(
                    self.kernel, 
                    input_channel[x-radius:x+radius+1, y-radius:y+radius+1]
                )
                normalized_value = self.get_normalized_output(convolution)
                output_img[x, y] = normalized_value

        return output_img 

    def blur(self, input_img: numpy.ndarray):
        """
        Function applies blurring effect to the image
        Allowed number of channels - (1 or 3 or 4)
        """
        if len(input_img.shape) == 2:
            return self.blur_channel(input_img)
        else:
            channels = self._split_channels(input_img)
            blurred_img = numpy.zeros(shape=input_img.shape)

            for idx, channel in enumerate(channels):
                blurred_channel = self.blur_channel(channel)
                blurred_img[:, :, idx] = blurred_channel
            return blurred_img