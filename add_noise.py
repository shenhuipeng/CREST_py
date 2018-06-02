from skimage.util import random_noise
import numpy as np


def add_random_noise(image):
    list ={0:"gaussian", 1:"localvar",2:"poisson" ,3:"salt" ,4:"pepper" ,5:"s&p",6:"speckle"}
    output = random_noise(image, mode=list.get(np.random.randint(1,7)))

    return output
