import numpy as np
from scipy.ndimage.filters import convolve1d


def ivarsmooth(flux, ivar, width):
      """
      Return the inverse-variance smoothed spectrum; width is the size of
      the smoothing window in pixels.
      """
      
      #  inspired by https://github.com/themiyan/astronomy
      weights = np.ones(int(width))
      convolved_numerator = convolve1d(flux * ivar , weights, mode='constant')
      convolved_denominator = convolve1d(ivar, weights, mode='constant')
      smoothed_flux = convolved_numerator / convolved_denominator 

      return smoothed_flux
