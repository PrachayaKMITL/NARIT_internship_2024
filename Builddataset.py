import os 
import json
import pandas as pd
import numpy as np
import cv2
from preprocessing import *
from TotalCalculation import *
sky_cam = str(input("Enter sky camera location(Astropark/Chile/China) : "))
with open("configuration.json", 'r') as config_file:
    config = json.load(config_file)
path = input("Enter folder path : ")
location = config[sky_cam]['parameters']['location']
time_zone = config[sky_cam]['parameters']['timezone']
mask_path = r'mask_delete_5.png'
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
mask = mask[(int(mask.shape[0]/2)-270):int((mask.shape[0]/2)+270),int((mask.shape[1]/2)-290):int((mask.shape[1]/2)+280)]
properties = ['contrast',
            'dissimilarity',
            'homogeneity',
            'energy',
            'correlation',
            'ASM']
day = 1
folders = os.listdir(path)
start_date = str(dt(2024,1,1).date())
for folder in folders:
    folder = os.path.join(path,folder)
    images,filename = preprocessData().load_images_and_preprocess(folder,mask=mask,apply_crop_sun=True)
    sunrise,sunset = SunPosition().SunriseSunset(filename=filename[50],location=location,Time_zone=time_zone,start_date=start_date,include_end_date=True)
    masked,value,statisical = thresholding().RBratio(input=images,filename=filename,sunrise=sunrise,sunset=sunset,Time_zone=time_zone)
    #grad = preprocessData().Edging(input=masked,ker_size=7,cliplimit=40,gridsize=14,bias=50)
    glcm = preprocessData().computeGlcm(image=masked,distance=[3],angle=[45])
    df = preprocessData().getDataframe(properties,glcm,index=filename,intensity=value,statistical=statisical)
    output_filename = f'GLCM_ALL_sky_{day}_dis3_ang45.csv'
    output_path = os.path.join(r'C:\Users\ASUS\Documents\NARIT_internship_data\CSV_dastaset_old\Dataset_06_2024', output_filename)
    print(f"---File {day} write complete---")
    day = day+1
    df.to_csv(output_path)
