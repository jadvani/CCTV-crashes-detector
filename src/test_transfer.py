# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:14:37 2020

@author: Javier
"""
import cv2
import os
def resize_image_75(img_path, shape=75):
    img = cv2.imread(img_path)
    return cv2.resize(img, dsize=(shape, shape), interpolation=cv2.INTER_CUBIC)


def resize(path):
    dirs = os.listdir(path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = resize_image_75(img_path=path+item)
            f, e = os.path.splitext(path+item)
            name=f.split('/dataset/')[0] +'/dataset/resize/'+ f.split('/dataset/')[1]+ '.jpg'
            print(name)
            cv2.imwrite(name,im)

resize("F:/TFM_datasets/car-crashes-detector/dataset/train/1/")
resize("F:/TFM_datasets/car-crashes-detector/dataset/train/2/")
resize("F:/TFM_datasets/car-crashes-detector/dataset/test/1/")
resize("F:/TFM_datasets/car-crashes-detector/dataset/test/2/")
resize("F:/TFM_datasets/car-crashes-detector/dataset/val/1/")
resize("F:/TFM_datasets/car-crashes-detector/dataset/val/2/")

#%%
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (75, 75, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

#%%

from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir = 'F:/TFM_datasets/car-crashes-detector/dataset/resize'
train_dir = os.path.join( base_dir, 'train')
validation_dir = os.path.join( base_dir, 'val')


train_cats_dir = os.path.join(train_dir, '1') # Directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, '2') # Directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, '1') # Directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, '2')# Directory with our validation dog pictures

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (75, 75))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (75, 75))

#%%

history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 20,
            validation_steps = 50,
            verbose = 2)
model.save_weights('transfer_inceptionv3.h5')

#%%

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.savefig('result_accuracy.png')