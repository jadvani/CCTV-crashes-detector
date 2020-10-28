# import usefull libraries
import numpy as np
import cv2

# Set global parameters
RED = 0
GREEN = 1
BLUE = 2


import cv2 

import matplotlib.pyplot as plt
img = cv2.imread("preview.jpg")

term =100
for element in img:
    for y in element:
        if(y[GREEN]+term<=255):
            y[GREEN]=y[GREEN]+term 
        else:
            y[RED]=255
plt.imshow(img[...,::-1])
 