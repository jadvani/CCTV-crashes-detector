# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 07:11:33 2020

@author: Javier
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np

def predict(img_path, model_path='model.h5',img_size=28):
    model = load_model(model_path)
    img_loaded = image.load_img(img_path,False,target_size=(img_size,img_size))
    x = image.img_to_array(img_loaded)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    return True if result[0][0] >= 0.5 else False


