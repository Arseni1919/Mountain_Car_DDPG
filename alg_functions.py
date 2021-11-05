from GLOBALS import *


def moving_average(x, w):
    w = int(w)
    return np.convolve(x, np.ones(w), 'valid') / w