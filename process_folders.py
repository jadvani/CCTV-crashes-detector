# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:06:05 2020

@author: Javier
"""
from glob import glob
import process_folder
dataset_directory = 'F:\\TFM_datasets\\extracted_frames\\*\\'
crash_directories = glob(dataset_directory)

for directory in crash_directories:
    print("processsing folder: "+directory)
    process_folder(directory)
    

