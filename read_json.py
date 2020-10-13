# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:23:01 2020

@author: Javier
"""
import json
import cv2
# Opening JSON file 
f = open('test.json',) 
  
# returns JSON object as  
# a dictionary 
data = json.load(f) 
  
# Iterating through the json 
# list 

# Closing file 
f.close() 

import matplotlib.pyplot as plt

def onclick(event):
    global offset
    offset = event.ydata
    fig.canvas.mpl_disconnect(cid)
    plt.close()
    return




for element in data:
    img = cv2.imread("F:\\TFM_datasets\\extracted_frames\\" + element["folder"] + "\\" + str(element["start"]) + ".jpg")
    # print()
   # print("F:\\TFM_datasets\\extracted_frames\\" + element["folder"] + "\\"+str(element["end"]) +".jpg")
    fig = plt.figure() 
    plt.show(img)
    plt.title('Mouse left-click on the desired offset')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    while not 'offset' in locals():
        plt.pause(5)
    print('Offset =', offset)
  