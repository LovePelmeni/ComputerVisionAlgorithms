import numpy

class SSIM(object):
    """
    Implementation of the Similarity Structural Index Measure (SSIM)
    
    Parameters:
    -----------
    alpha - weight for luminance 
    beta - weight for contrast
    gamma - weight for structure (detail similarity)

    all sums up to 1.
    """
    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 

    def compute(self, orig_img: numpy.ndarray, comp_img: numpy.ndarray):

        dy_range = numpy.ptp(orig_img)
        c1, c2, c3 = self._compute_constants(dy_range=dy_range)

        l_comp = self._compute_luminance_diff(orig_img, comp_img, c1) ** self.alpha
        c_comp = self.compute_contrast_diff(orig_img, comp_img, c2) ** self.beta 
        s_comp = self.compute_structure_diff(orig_img, comp_img, c3) ** self.gamma
        
        return l_comp * c_comp * s_comp

    def _compute_constants(self, dy_range: float):
        c1 = (0.01 * dy_range) ** 2
        c2 = (0.03 * dy_range) ** 2
        c3 = c2 / 2
        return c1, c2, c3

    def _compute_luminance_diff(self, img1, img2, c1):
        mean_x = numpy.mean(img1.flatten())
        mean_y = numpy.mean(img2.flatten())
        return (2 * mean_x * mean_y + c1) / ((mean_x ** 2) + (mean_y ** 2) + c1) 

    def _compute_contrast_diff(self, img1, img2, c2):
        sigma_x = numpy.std(img1.flatten())
        sigma_y = numpy.std(img2.flatten())
        return (2 * sigma_x * sigma_y + c2) / ((sigma_x ** 2) + (sigma_y ** 2) + c2)

    def _compute_structure_diff(self, img1, img2, c3):
        cov_xy = numpy.cov(img1.flatten(), img2.flatten())
        sigma_x = numpy.std(img1.flatten())
        sigma_y = numpy.std(img2.flatten())
        return (cov_xy + c3) / (sigma_x * sigma_y + c3)

        

