# Accidents Detection on CCTV images. 

 Detect if an abnormal action happens given a CCTV video sequence, using Computer Vision and Convolutional Neural Networks. These code scripts are part of my Final Masters Project for the University of Seville. Special thanks to Ignacio Pérez-Herrera and Miguel Ángel Martínez del Amor for all the support given.  
<p align="center">
  <img src="https://github.com/jadvani/car-crashes-detector/blob/master/img/preview.png">
</p>

<br>

## Code structure

The source code is divided in four main folders:

* [detection_framework](https://github.com/jadvani/CCTV-crashes-detector/tree/master/src/detection_framework), which contains the final version of the detection tool. 
* [train_cnn](https://github.com/jadvani/CCTV-crashes-detector/tree/master/src/train_cnn) for all the scripts used during the training. Please, note that these scripts have been constantly being modified and might not contain the architecture selected for the model integrated in the framework. For further information about the different configurations selected, please read the Final Thesis Project [here]().
* [automation](https://github.com/jadvani/CCTV-crashes-detector/tree/master/src/automation) includes all the actions needed to agilize the creation of the datasets, data conversion or automatic downloads. 
* [pocs](https://github.com/jadvani/CCTV-crashes-detector/tree/master/src/pocs) are the different proofs of concept done in different parts of the code. All the valuable code from these scripts has been included in the final detection framework. 

## Datasets

The first dataset used to train the CNNs involved in this solution was taken from this repository of [car damages](https://github.com/mghatee/Accident-Images-Analysis-Dataset). A copy of this dataset has been included here. 

Folder 1 includes 2500 images with label "without-accident".
Folder 2 includes 2398 images with label "with-accident".

A second dataset of images has been created from the original [CADP dataset](https://ankitshah009.github.io/accident_forecasting_traffic_camera), taking multiple image parts and relabelling them. The final version of this new Dataset will be shared in Kaggle. 

## Train and test the model

1. Once you download the two-classes dataset, generate the train/test/val split using the script src/automation/train_test_split.py

2. Run the car-crashes-detector.py to train the model.

3. Predict a single image with src/pocs/predict-single-image.py


## Run the complete framework solution

1. Download at least one of the accident directories from CADP. 
2. src/detection_framework/process_folder.py will run all the detection steps needed to perform the classification: OpenCV + YOLO + CNN
3. The result of the execution shows the abnormal image crops taken from the whole video recording. 
