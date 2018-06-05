from skimage.util import random_noise,img_as_ubyte
import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_random_noise(image):
    list ={0:"gaussian", 1:"localvar",2:"poisson" ,3:"salt" ,4:"pepper" ,5:"s&p",6:"speckle"}
    output = img_as_ubyte(random_noise(image, mode=list.get(np.random.randint(1,7))))
    fig = plt.figure()
        
    ax = fig.add_subplot(111)
    ax.imshow(output)
    plt.show()

    return output
