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
process = opencv_processor('F:\\TFM_datasets\\extracted_frames\\000079',interval=2, threshold=30, dilation=Dilation.HIGH)
process.process_folder()

# OpenCV
crashes = process.possible_crash_sections

#YOLO 
final_crashes = []

for possible_crash in crashes:
    yolo = yolo_detector("C:\\Users\\Javier\\Downloads\\darknet-master\\cfg",0.2,0.3)
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
            if((res.shape[0]>0) and (res.shape[1]>0)): # si la imagen está vacía, al tratar de ver el tamaño salta un error
                final_crashes.append(res) #intersecciones finales que van a ser procesadas con red neuronal
        except:
            pass
        
      
for crash in final_crashes:
    cv2.imwrite('evaluate_image.jpg',crash)
    predict_yolo_image.predict('evaluate_image.jpg')    
