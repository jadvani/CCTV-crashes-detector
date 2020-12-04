# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:25:11 2020

@author: Javier
"""

from pytube import YouTube



def read_video_urls():
    with open('videos.txt') as urls:
        return urls.read().splitlines() 
    
    
video_urls = read_video_urls()
for url in video_urls:
    print("Downloading video "+str(video_urls.index(url))+" from "+str(len(video_urls)))
    YouTube(url).streams[0].download(output_path='F:\\TFM_datasets\\car-crashes-detector\\videos')
