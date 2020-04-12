# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:06:52 2020

@author: Javier
"""

from PIL import Image
from skimage import transform
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# force a resized image load 
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (28, 28, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image



# dimensions of our images    -----   are these then grayscale (black and white)?
img_width, img_height = 313, 220

# load the model we saved
model = load_model('model.h5')

# Get test image ready
test_image = image.load_img('dataset/val/2/1978.jpg', target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

test_image = test_image.reshape(img_width, img_height*3)    # Ambiguity!
# Should this instead be: test_image.reshape(img_width, img_height, 3) ??

result = model.predict(test_image, batch_size=1)
print(result)