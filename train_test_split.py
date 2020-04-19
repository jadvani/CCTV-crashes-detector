# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:09:57 2020

@author: Javier
"""
import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'F:/TFM_datasets/car-crashes-detector/dataset'
posCls = '/1'
negCls = '/2'

def remove_folder_if_exists(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

elements = [posCls, negCls]

remove_folder_if_exists(root_dir +'/train')
remove_folder_if_exists(root_dir +'/val')
remove_folder_if_exists(root_dir +'/test')

os.makedirs(root_dir +'/train' + posCls)
os.makedirs(root_dir +'/train' + negCls)
os.makedirs(root_dir +'/val' + posCls)
os.makedirs(root_dir +'/val' + negCls)
os.makedirs(root_dir +'/test' + posCls)
os.makedirs(root_dir +'/test' + negCls)

# Creating partitions of the data after shuffeling
for currentCls in elements:
    src = root_dir+currentCls # Folder to copy images from
    
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
    
    
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))
    
    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir+"/train"+currentCls)
    
    for name in val_FileNames:
        shutil.copy(name, root_dir+"/val"+currentCls)
    
    for name in test_FileNames:
        shutil.copy(name, root_dir+"/test"+currentCls)