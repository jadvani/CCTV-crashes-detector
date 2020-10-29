# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:22:32 2020

@author: Javier
"""
a=(209, 71, 47, 37)
b=(209, 71, 52, 40)

def number_in_range(num1,num2, ran=5):
    return (abs(num1-num2)<=ran and abs(num1-num2)>=0)

def similar_tuples(a,b):
    return number_in_range(a[0],b[0]) and number_in_range(a[1],b[1]) and number_in_range(a[2],b[2]) and number_in_range(a[3],b[3])

print(similar_tuples(a,b))