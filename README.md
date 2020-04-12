# car-crashes-detector
 Detect if a car is damaged using CNN. Final Masters Project for the University of Seville.  

![preview](https://github.com/jadvani/car-crashes-detector/blob/master/preview.jpg)

## Dataset

The dataset used has been taken from this repository https://github.com/mghatee/Accident-Images-Analysis-Dataset
<br>

According to the original dataset:

Folder 1 includes 2500 images with label "without-accident".
Folder 2 includes 2398 images with label "with-accident".

## How to run the code

1. Generate the train/test/val split using train_test_split.py

Questions:

We have a different number of image types in our dataset. Do I remove some of the "without-accident" ones in order to have the same number of class samples or not?

