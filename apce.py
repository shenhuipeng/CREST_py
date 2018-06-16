
# CVPR2017
# Large Margin Object Tracking with Circulant Feature Maps
# average peak-tocorrelation-energy
import numpy as np
def  get_APCE(map):
    h, w = map.shape
    max_F = np.max(map)
    min_F = np.min(map)
    apce = np.square(max_F-min_F) / (np.sum(np.square(map[:,:]-min_F))/(h * w))
    return apce


