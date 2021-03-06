# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:59:15 2020

@author: Javier
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets  import RectangleSelector
import cv2
import pyperclip

xdata = np.linspace(0,9*np.pi, num=301)
ydata = np.sin(xdata)

fig, ax = plt.subplots()
line, = ax.plot(xdata, ydata)
img=cv2.imread("F:\\TFM_datasets\\extracted_frames\\001019\\81.jpg")



def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    coords = "["+str(int(x1))+","+str(int(y1))+","+str(int(x2-x1))+","+str(int(y2-y1))+"]"
    print(coords)
    pyperclip.copy(coords)
    ax.add_patch(rect)


rs = RectangleSelector(ax, line_select_callback,
                       drawtype='box', useblit=False, button=[1], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)

plt.imshow(img[...,::-1])
plt.pause(8)
plt.close()