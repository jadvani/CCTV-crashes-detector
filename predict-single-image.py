# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:06:52 2020

@author: Javier
"""


from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
test_model = load_model('model.h5')
img = image.load_img('dataset/val/1/133.jpg',False,target_size=(28,28))
#img = image.load_img('dataset/val/2/1978.jpg',False,target_size=(img_width,img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
result = test_model.predict(x)
if result[0][0] >= 0.5:
    prediction = '2 (with accident)'
else:
    prediction = '1 (without accident)'
print(prediction)