# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:25:05 2020

@author: Javier
"""
import os 
import cv2
import re
import matplotlib.pyplot as plt

def read_frames(frames_path):
    col_frames = [f for f in os.listdir(frames_path) if (f.endswith('.jpg') or f.endswith('.png'))] #quitar ficheros innecesarios
    col_frames.sort(key=lambda f:int(re.sub('\D', '', f)))
    image_dict = dict()
       
    for i in col_frames:
        img=cv2.imread(frames_path+"\\"+i)
        try:
            height=img.shape[0]
            width=img.shape[1]
            image_dict[i]=img
        except:
            print("image",i, "is corrupted.")
        
        
    return image_dict

col_images = read_frames('F:\\TFM_datasets\\extracted_frames\\000001')
# img = col_images['701.jpg'] 
# cv2.rectangle(img, (141, 526), (75+141, 526+28), (255,0,0), 2) # las áreas de annotations no corresponden con el accidente
# plt.imshow(img)