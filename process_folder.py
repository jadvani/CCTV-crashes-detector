# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:11:06 2020

@author: Javier
"""
# Calling OpenCV and Yolo classes here. 
from opencv_preprocessing import opencv_processor
from dilation import Dilation
from yolo_detector import yolo_detector
import cv2
import matplotlib.pyplot as plt
import predict_yolo_image

import os 

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def process_folder(path):
    process = opencv_processor(path ,interval=3, threshold=30, dilation=Dilation.HIGH,show_images=False)
    process.process_folder()
    folder_name=path.split("\\")[-1]
    
    # OpenCV
    crashes = process.possible_crash_sections
    
    #YOLO 
    final_crashes = []
    yolo = yolo_detector("C:\\Users\\Javier\\Downloads\\darknet-master\\cfg",0.2,0.3)
    for possible_crash in crashes:
        yolo.print_coco_names_folderpath()
        img,boxes,ids=yolo.process_image(possible_crash)
        yolo.get_union_areas(boxes)
        potential_crashes=yolo.potential_crashes
        i = 0
        for coord in yolo.coord_unions:
            print(i)
            org_img = yolo.original_image
            # res=cv2.rectangle(org_img, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (255, 0, 0), 2)
            #final_crashes.append(res)
            try:
                res = org_img[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
    
                i=i+1
                if((res.shape[0]>0) and (res.shape[1]>0)): # si la imagen no está vacía
                    final_crashes.append(res) #se procesará
            except:
                pass
    # CNN   
    index=0      
    create_folder('results_processing')
    for crash in final_crashes:
        create_folder('results_processing\\'+folder_name)
        image_path='results_processing\\'+folder_name+"\\"+folder_name+"_"+str(index)+'.jpg'
        print(image_path)
        cv2.imwrite(image_path,crash)
        is_accident=predict_yolo_image.predict(img_path=image_path)  
        index=index+1
        # if(is_accident):
        #     plt.imshow(crash)
        #     plt.pause(1)
        #     plt.close()
process_folder(path = 'F:\\TFM_datasets\\extracted_frames\\000001')