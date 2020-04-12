# car-crashes-detector
 Detect if a car is damaged or not, using CNN. This is part of my Final Masters Project for the University of Seville, thanks to the support of .  

![preview](https://github.com/jadvani/car-crashes-detector/blob/master/preview.jpg)

## Dataset

The dataset used has been taken from this repository https://github.com/mghatee/Accident-Images-Analysis-Dataset
<br>

According to the original dataset:

Folder 1 includes 2500 images with label "without-accident".
Folder 2 includes 2398 images with label "with-accident".

## How to run the code

1. Generate the train/test/val split using train_test_split.py

2. Run the car-crashes-detector.py to train the model. It currently gets an accuracy: 0.8954 - val_loss: 0.1807 - val_accuracy: 0.8693 (aprox)

3. Predict a single image with predict-single-image.py

We have a different number of image types in our dataset. Do I remove some of the "without-accident" ones in order to have the same number of class samples or not?

