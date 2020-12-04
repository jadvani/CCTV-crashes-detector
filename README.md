# Accidents Detection on CCTV images. 

 Detect if an abnormal action happens given a CCTV video sequence, using Computer Vision and Convolutional Neural Networks. These code scripts are part of my Final Masters Project for the University of Seville. Special thanks to Ignacio Pérez-Herrera and Miguel Ángel Martínez del Amor for all the support given.  

![preview](https://github.com/jadvani/car-crashes-detector/blob/master/preview.jpg)

## Dataset

The dataset used has been taken from this repository https://github.com/mghatee/Accident-Images-Analysis-Dataset
<br>

According to the original dataset:

Folder 1 includes 2500 images with label "without-accident".
Folder 2 includes 2398 images with label "with-accident".

# How to prepare the enviroment

1. Based on the original [CADP dataset](https://ankitshah009.github.io/accident_forecasting_traffic_camera), we have created a new dataset. TODO: upload dataset to kaggle. 
2. The script process_folder.py runs all the block dependencies involved in the accident detection. Please, note that we are using Python 3.8 and Keras API from Tensorflow to run this code. 

Further details can be found on the published documente [here] TODO:insert link

## How to run the training script 

1. Generate the train/test/val split using train_test_split.py

2. Run the car-crashes-detector.py to train the model.

3. Predict a single image with predict-single-image.py

We have a different number of image types in our dataset. Do I remove some of the "without-accident" ones in order to have the same number of class samples or not?

