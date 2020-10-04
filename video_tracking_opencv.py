# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:12:26 2020

@author: Javier
"""

import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.spatial import distance

def get_countour_centers(countour_array):
    centers = []
    for elem in countour_array:
        x=elem[0]+int(elem[2]/2)
        y=elem[1]+int(elem[3]/2)
        centers.append([x,y])
    return centers
#leer frames de una carpeta
def read_frames(frames_path):
    col_frames = [f for f in os.listdir(frames_path) if (f.endswith('.jpg') or f.endswith('.png'))] #quitar ficheros innecesarios
    print(len(col_frames))
    col_frames.sort(key=lambda f:int(re.sub('\D', '', f)))
    col_images=[]
    
    for i in col_frames:
        col_images.append(cv2.imread(frames_path+"\\"+i))
        
    return col_images

# leer una imagen como escala de grises
def read_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_centers(centers,dmy):
    red = [255,0,0]
    for center in centers:
        dmy[center[1],center[0]]=red
    return dmy

def diff_frames(img1, img2):
    grayA = read_gray(img1)
    grayB = read_gray(img2)
    
    #diferencia entre imagenes 
    diff_image = cv2.absdiff(grayB, grayA)
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    
    # dilatacion
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh,kernel,iterations = 1)
    
    #detectamos contornos obtenidos en los dif frames
    contours, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    valid_cntrs = []
    final_countours = []
    
    for i,cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr)
        valid_cntrs.append(cntr)
        final_countours.append([x,y,w,h])
    
    len(valid_cntrs)
    return img1.copy(), valid_cntrs, final_countours

col_images = read_frames('F:\\TFM_datasets\\extracted_frames\\000005')
[dmy, valid_cntrs, final_countours]=diff_frames(col_images[105],col_images[106])
centers=get_countour_centers(final_countours)
dmy = draw_centers(centers, dmy)
cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)

plt.imshow(dmy)

print(centers)

# representamos distancias entre coches en el transcurso de 2 fotogramas. 
def euclidean_matrix(centers):
    euclidean_distances = []
    for i in range(len(centers)):
        for j in range(len(centers)):
            euclidean_distances.append(round(distance.euclidean(centers[i],centers[j]),2))
    return euclidean_distances
            
#print("distancia euclidea: ",round(distance.euclidean(centers[0],centers[1]),2))
ED = np.array(euclidean_matrix(centers))
print(ED.reshape((len(centers),len(centers))))

plt.show()

