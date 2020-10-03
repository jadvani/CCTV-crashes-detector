# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:22:38 2020

@author: Javier
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sinerelu import SineReLU
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from keras.applications.vgg16 import VGG16, decode_predictions

modeloVGG16 = VGG16()
#%%

# dimensions of our images.
img_width, img_height = 28, 28

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'
nb_train_samples = 1678
nb_validation_samples = 360
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    


#%%
"""

SineReLU Activation Function


Z                              Z>0
ϵ(sin(Z) - cos(Z))       Z<=0


https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d
"""    
    
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation(SineReLU()))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation(SineReLU()))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation(SineReLU()))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation(SineReLU()))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#%%

#https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
# Metrics have been removed from Keras core. We need to calculate accuracy F1, precission & recall manually.
# tp / (tp + fn)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
# tp / (tp + fp)
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
# F-1 = 2 * (precision * recall) / (precision + recall)
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#%%
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', precision_m,f1_m,recall_m])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
#%%
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
#Y_pred = model.predict_generator(validation_generator, 2*nb_validation_samples // batch_size+1)
#print('Confusion Matrix')
#print(confusion_matrix(validation_generator.classes, Y_pred))
#print('Classification Report')
#target_names = ['1', '2']
#print(classification_report(validation_generator.classes, Y_pred, target_names=target_names))
model.save_weights('first_try.h5')