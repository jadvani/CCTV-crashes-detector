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
import collections

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
            img.shape[0] # si la imagen está vacía, al tratar de ver el tamaño salta un error
            img.shape[1]
            col_images.append(img)
        except: 
            print("image",i, "is corrupted.")
        
    return col_images



def draw_centers(centers,dmy):
    red = [255,0,0]
    for center in centers:
        dmy[center[1],center[0]]=red
    return dmy


# representamos distancias entre 2 elementos móviles en el transcurso de 2 fotogramas como una matriz. 

#     1   2   3
# 1   0   X   Y
# 2   X   0   Z
# 3   Y   Z   0 
#
# Teniendo 3 vehículos / centroides, la matriz representa 
# las distancias entre cada uno de ellos. 
# La diagonal es 0 siempre, y tiene un tamaño igual al de número de centroides.

def euclidean_matrix(centers):
    euclidean_distances = []
    for i in range(len(centers)):
        for j in range(len(centers)):
            euclidean_distances.append(round(distance.euclidean(centers[i],centers[j]),2))
    return euclidean_distances


def is_sublist_in_list(pattern, list_coordinates):
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
        
def set_up_figure():
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect()
    fig.canvas.mpl_connect()
    return fig, ax

                        #print(ED.reshape((len(centers),len(centers))))
class opencv_processor():
    
    def __init__(self,col_images, interval=2):
        #read_frames('F:\\TFM_datasets\\extracted_frames\\000079')
        self.col_images = col_images
        self.interval = interval
        
        # leer una imagen como escala de grises
    def read_gray(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        
    # similar al juego de las diferencias, restamos dos fotogramas para ver qué es lo que cambia de un instante a otro.
    def diff_frames(self,img1, img2):
        grayA = self.read_gray(img1)
        grayB = self.read_gray(img2)
        
        #diferencia entre imagenes 
        diff_image = cv2.absdiff(grayB, grayA)
        ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
    
        
        # dilatacion
        kernel = np.ones((3,3),np.uint8)
        dilated = cv2.dilate(thresh,kernel,iterations = 1)
        
        #detectamos contornos obtenidos en los dif frames
        contours, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        final_countours = []
        
        for i,cntr in enumerate(contours):
            x,y,w,h = cv2.boundingRect(cntr)
            final_countours.append([x,y,w,h])
        
        return img1.copy(), contours, final_countours
    
    def detect_matrix_changes(self,dmy,final_countours,i,valid_cntrs,previous_centers,m):
        centers=get_countour_centers(final_countours)
        dmy = draw_centers(centers, dmy)
        cv2.drawContours(dmy, valid_cntrs, -1, (127,200,0), 2)
                
                                #print(centers)
                                #print("distancia euclidea: ",round(distance.euclidean(centers[0],centers[1]),2))
        ED = np.array(euclidean_matrix(centers))
        ED = ED.reshape((len(centers),len(centers)))
        print("Imágenes ",i,(i+(self.interval-1)),". La matriz ",m," tiene ",len(centers)," siluetas")
        if(previous_centers>len(centers) and previous_centers>0 and len(centers)>0):
            print("posible accidente!")
            dmy=draw_squared_accident(centers, dmy, ED)
        return dmy,centers
    
    def show_processed_image(self,i,dmy):
        txt = plt.text(10,10,str(i),horizontalalignment='center',verticalalignment='center')
        plt.draw()
        plt.imshow(dmy)
        plt.pause(0.1)
        plt.show()
        txt.remove()
        plt.draw()
        
    def process_folder(self):
        matrix_counter = 0  
        previous_centers=0
        for i in range(0,len(self.col_images),self.interval):
            
            if(i<len(self.col_images)-(self.interval-1)):
                [dmy, valid_cntrs, final_countours]=self.diff_frames(self.col_images[i],self.col_images[i+(self.interval-1)])
                [dmy,centers]=self.detect_matrix_changes(dmy,final_countours,i,valid_cntrs,previous_centers,matrix_counter)
                self.show_processed_image(i,dmy)
                matrix_counter = matrix_counter + 1 
                previous_centers = len(centers)
                
                
process = opencv_processor(col_images=read_frames('F:\\TFM_datasets\\extracted_frames\\000079'),interval=2)
process.process_folder()
