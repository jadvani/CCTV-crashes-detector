# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:37:32 2020

@author: Javier
"""
import pandas as pd
import ast
import cv2
import os
# import matplotlib.pyplot as plt
#%%
data_labelled = pd.read_excel('JSON\\CADP Dataset relabelled_27_10_2020.xlsx')
annotations = data_labelled[['folder', 'start', 'start_coord', 'end','end_coord']]
nonempty_annotations = annotations.dropna() # eliminamos anotaciones que no se hayan verificado aún
base_path = 'F:\\TFM_datasets\\extracted_frames\\'
#%%

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
def crop_images(annotation):
     folder=folder_name(annotation.folder)
     try:
         start_coords = ast.literal_eval(annotation.start_coord)
         end_coords = ast.literal_eval(annotation.end_coord)
         start_image = cv2.imread(base_path+folder+"\\"+str(int(annotation.start))+".jpg")[start_coords[1]:start_coords[1]+start_coords[3], start_coords[0]:start_coords[0]+start_coords[2]]
         end_image = cv2.imread(base_path+folder+"\\"+str(int(annotation.end))+".jpg")[end_coords[1]:end_coords[1]+end_coords[3], end_coords[0]:end_coords[0]+end_coords[2]]
         create_folder("JSON\\start")
         create_folder("JSON\\end")
         [start_path, end_path] = ("JSON\\start\\"+folder+"_"+str(int(annotation.start))+".jpg","JSON\\end\\"+folder+"_"+str(int(annotation.end))+".jpg")
         cv2.imwrite(start_path, start_image)   
         cv2.imwrite(end_path, end_image)  
         return start_image, end_image
     except:
        print("revisar carpeta",folder)

     
def folder_name(number):
    return "{:06d}".format(number)

#%%
i=1
for annotation in nonempty_annotations.itertuples():
    # print(i)
    i=i+1
    try:
        folder=folder_name(annotation.folder)
        print(base_path+folder+"\\"+str(int(annotation.start))+".jpg")
        start, end = crop_images(annotation)
        # plt.imshow(start[...,::-1])
        # plt.pause(1)
        # plt.close()
        # plt.imshow(end[...,::-1])
        # plt.pause(1)
        # plt.close()
    except:
        pass

    
