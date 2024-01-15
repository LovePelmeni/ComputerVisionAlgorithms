import numpy
class BilateralFilter(object):

    def __init__(self, 
        sigmaSpaceX: float, 
        sigmaColor: float,
        kernel_size: int,
        sigmaSpaceY: float = None,
    ):
        self.sigmaSpaceX = sigmaSpaceX 
        self.sigmaSpaceY = sigmaSpaceY if sigmaSpaceY else sigmaSpaceX
        self.sigmaColor = sigmaColor
        self.kernel_size = kernel_size
        self.gaussian_kernel = self._generate_gaussian_kernel(kernel_size)

    def _generate_gaussian_kernel(self, kernel_size: int):
        """
        Function generates gaussian kernel
        """
        if kernel_size % 2 == 0:
            raise ValueError(msg='kernel size should be odd, not even')

        output_kernel = numpy.zeros(shape=(self.kernel_size, self.kernel_size))
        radius = (self.kernel_size // 2)

        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                output_kernel[x, y] = self.gaussian_eq(
                    posX=x, 
                    posY=y, 
                    posx0=radius, 
                    posy0=radius
                )
        return output_kernel
    
    def gaussian_eq(self, posX: int, posY: int, posx0: int, posy0: int):
        """
        Function computes the spatial distance between 2 pixels
        belonging to the same kernel window
        """
        x_part = numpy.power((posX - posx0), 2) / (2 * numpy.power(self.sigmaSpaceX, 2))
        y_part = numpy.power((posY - posy0), 2) / (2 * numpy.power(self.sigmaSpaceY, 2))
        exp_power = - (y_part + x_part)
        return (1 / 2 * numpy.pi * self.sigmaSpaceX * self.sigmaSpaceY) * numpy.exp(exp_power)

    def _generate_intensity_kernel(self, input_region: numpy.ndarray):
        """
        Function computes intensity distance between 2 pixels

        Parameters:
        -----------
        center_intensity (int) - center pixel intensity
        pos_intensity (int) - intenisty of the neighbor pixel
        """
        output_intensities = numpy.zeros_like(a=input_region)
        height, width = input_region.shape
        centerX = centerY = self.kernel_size // 2
        
        for x in range(height):
            for y in range(width):
                exp_power = -(((input_region[x, y] - input_region[centerX, centerY]) ** 2) / (2 * numpy.power(self.sigmaColor, 2)))
                output_intensities[x, y] = numpy.exp(exp_power)
        return output_intensities

    def blur(self, input_img: numpy.ndarray):

        output_img = numpy.zeros_like(a=input_img)
        height, width = input_img.shape 
        center_coord = self.kernel_size // 2

        for x in range(center_coord, height - center_coord):
            for y in range(center_coord, width - center_coord):

                input_region = input_img[
                    x-center_coord:x+center_coord+1, 
                    y-center_coord:y+center_coord+1
                ]
                
                intensity_kernel = self._generate_intensity_kernel(input_region=input_region)

                general_kernel = numpy.multiply(self.gaussian_kernel, intensity_kernel)
                
                convolution_output = numpy.sum(
                    numpy.multiply(
                        general_kernel,
                        input_region
                    )
                ) / numpy.sum(general_kernel)

                # normalizing output by it's weights
                output_img[x, y] = convolution_output

        return output_img.astype(numpy.uint8)

