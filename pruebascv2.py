# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:33:15 2020

@author: Javier
"""
import cv2
import matplotlib.pyplot as plt
im1 = cv2.imread("F:\\TFM_datasets\\extracted_frames\\001084\\32.jpg")
im2 = cv2.imread("F:\\TFM_datasets\\extracted_frames\\001084\\38.jpg")

def read_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


im1_gray = read_gray(im1)
im2_gray = read_gray(im2)

plt.imshow(im2_gray,cmap='gray')
diff_image = cv2.absdiff(im1_gray, im2_gray)
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

plt.imshow(thresh,cmap='gray')
