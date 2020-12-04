# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 00:49:50 2020

@author: Javier
"""
import cv2
import os
import re
import time
col_images=[]
def read_frames(frames_path):
    frames = [f for f in os.listdir(frames_path) if (f.endswith('.jpg') or f.endswith('.png'))] #quitar ficheros innecesarios
    frames.sort(key=lambda f:int(re.sub('\D', '', f)))
    print(len(frames))
    for frame in frames:
        try:
            img = cv2.imread(frames_path+"\\"+frame)
            img.shape[0] # si la imagen está vacía, al tratar de ver el tamaño salta un error
            img.shape[1]
            col_images.append(img)
        except: 
            print("image",frame, "is corrupted.")
start_time = time.time()
read_frames('F:\\TFM_datasets\\extracted_frames\\000001')     
print("--- %s seconds ---" % (time.time() - start_time))