# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:12:26 2020

@author: Javier
"""

import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from operator import itemgetter 

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
    col_frames.sort(key=lambda f:int(re.sub('\D', '', f)))
    col_images=[]
    
    for i in col_frames:
        try:
            img = cv2.imread(frames_path+"\\"+i)
            height=img.shape[0]
            width=img.shape[1]
            col_images.append(img)
        except: 
            print("image",i, "is corrupted.")
        
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

# representamos distancias entre coches en el transcurso de 2 fotogramas. 
def euclidean_matrix(centers):
    euclidean_distances = []
    for i in range(len(centers)):
        for j in range(len(centers)):
            euclidean_distances.append(round(distance.euclidean(centers[i],centers[j]),2))
    return euclidean_distances

import collections

def is_sublist_in_list(pattern, list_coordinates):
    # Using Counter 
    inside = False
    for elem in list_coordinates: 
        if collections.Counter(elem) == collections.Counter(pattern) : 
            inside=True
          
    return inside

def get_max_min_tuples(tuples_list):
    return [min(tuples_list, key = itemgetter(0))[0], max(tuples_list, key = itemgetter(0))[0],
            min(tuples_list, key = itemgetter(1))[1],max(tuples_list, key = itemgetter(1))[1]]
            
def draw_squared_accident(centers, img, ED):
    final = []
    
    for i in range(0,len(centers)):
        similar_centers = []
        for j in range(0, len(centers)):
            if(ED[i,j]>0 and ED[i,j]<25):
                if(not is_sublist_in_list(centers[i], similar_centers)):
                    similar_centers.append(centers[i])
        if(len(similar_centers)>0):
            final.append(tuple(similar_centers[0]))
            min_max=get_max_min_tuples(final)
    if(len(similar_centers)>0):
        return cv2.rectangle(img, (min_max[0], min_max[2]), (min_max[1], min_max[3]), (255, 0, 0), 2)
    else: 
        return img
        
   

col_images = read_frames('F:\\TFM_datasets\\extracted_frames\\000051')
m = 0

# import sys
# sys.stdout=open("test.txt","w")
previous_centers=0
for i in range(0,len(col_images),2):
    
    if(i<len(col_images)-1):
        [dmy, valid_cntrs, final_countours]=diff_frames(col_images[i],col_images[i+1])
        centers=get_countour_centers(final_countours)
        dmy = draw_centers(centers, dmy)
        cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)

        
        
        #print(centers)
        #print("distancia euclidea: ",round(distance.euclidean(centers[0],centers[1]),2))
        ED = np.array(euclidean_matrix(centers))
        ED = ED.reshape((len(centers),len(centers)))
        print("Imágenes ",i,i+1,". La matriz ",m," tiene ",len(centers)," siluetas")
        if(previous_centers>len(centers) and previous_centers>0 and len(centers)>0):
            print("posible accidente!")
            dmy=draw_squared_accident(centers, dmy, ED)
        #print(ED.reshape((len(centers),len(centers))))
        plt.imshow(dmy)
        m = m + 1 
        previous_centers = len(centers)
        plt.pause(0.1)
        
        #plt.show()
# sys.stdout.close()

