# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 23:17:23 2020

@author: Javier
"""
from operator import itemgetter 
#import video_tracking_opencv
import cv2
import matplotlib.pyplot as plt
import numpy as np
list_test=[[201, 350], [197, 348], [197, 342], [253, 323], 
           [338, 319], [246, 320], [354, 309], [216, 303], 
           [229, 303], [335, 294], [228, 306], [233, 310], 
           [218, 281], [220, 271], [217, 264], [315, 263], 
           [227, 260], [326, 256], [198, 244], [226, 250], 
           [205, 242], [276, 188], [276, 188], [173, 184], 
           [164, 186], [273, 181], [239, 174], [142, 185], 
           [104, 171], [358, 171], [136, 170], [102, 165], 
           [143, 166], [127, 158], [347, 155], [338, 154], 
           [173, 164], [212, 151], [225, 147], [177, 148], 
           [270, 168], [218, 170], [266, 166], [261, 165], 
           [239, 152]]
tuples_test=[]
for elem in list_test:
    tuples_test.append(tuple(elem))
max_x = max(tuples_test, key = itemgetter(0))[0]
min_x = min(tuples_test, key = itemgetter(0))[0]

max_y = max(tuples_test, key = itemgetter(1))[1]
min_y = min(tuples_test, key = itemgetter(1))[1]


img = cv2.imread("F:/TFM_datasets/extracted_frames/000007/96.jpg")
cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
plt.imshow(img)