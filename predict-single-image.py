# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:06:52 2020

@author: Javier
"""


from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
test_model = load_model('model.h5')
img = image.load_img('dataset/test/1/131.jpg',False,target_size=(28,28))
#img = image.load_img('dataset/test/2/1978.jpg',False,target_size=(28,28))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
result = test_model.predict(x)
if result[0][0] >= 0.5:
    prediction = '2 (with accident)'
else:
    prediction = '1 (without accident)'
print(prediction)

#%%
from keras.models import load_model
from keras.preprocessing import imaçge 
from keras.applications.vgg16 import VGG16, decode_predictions
import numpy as np
import PIL
import matplotlib.pyplot as plt
modeloVGG16 = VGG16()


def predict(image_path):
    # Carga y redimensiona la imagen usando PIL.
    img = PIL.Image.open(image_path)
    #input_shape = model.layers[0].output_shape[1:3]
    img_resized = img.resize((224,224), PIL.Image.LANCZOS)

    # Dibuja la imagen.
    plt.imshow(img_resized)
    plt.show()

    # Convierte la imagen PIL a un numpy-array con la forma (shape) apropiada.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Usa el modelo VGG16 para hacer la predicción.
    # Esto devuelve un array con 1000 números, correspondientes a
    # las clases del dataset ImageNet.
    pred = modeloVGG16.predict(img_array)
    
    # Decodifica la salida del modelo VGG16.
    pred_decoded = decode_predictions(pred)[0]

    # Imprime las predicciónes.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))
predict(image_path='F:/TFM_datasets/car-crashes-detector/preview.jpg')