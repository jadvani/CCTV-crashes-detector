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

process = opencv_processor('F:\\TFM_datasets\\extracted_frames\\000079',interval=2, threshold=30, dilation=Dilation.HIGH)
process.process_folder()

crashes = process.possible_crash_sections

yolo = yolo_detector("C:\\Users\\Javier\\Downloads\\darknet-master\\cfg",0.2,0.3)
yolo.print_coco_names_folderpath()
final_crashes = []
res = []
for possible_crash in crashes:

    img,boxes, ids=yolo.process_image(possible_crash)
    yolo.get_union_areas(boxes)
    potential_crashes=yolo.potential_crashes
    i = 0
    for coord in yolo.coord_unions:
        print(i)
        res=cv2.rectangle(yolo.original_image, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (255, 0, 0), 2)
        #final_crashes.append(res)
        plt.imshow(res)
        plt.pause(2)
        i=i+1
        
        
    
