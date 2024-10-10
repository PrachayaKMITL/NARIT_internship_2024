import os 
import json
import pandas as pd
import numpy as np
import cv2

from src.preprocessing import *
from src.TotalCalculation import *
sky_cam = str(input("Select images to train(Astropark : (December,April,June)) : "))
with open("Configuration\\Dataset_configuration.json", 'r') as config_file:
    config = json.load(config_file)
path = input("Enter folder path : ")
GLCM_param = input("Enter distance and angle in list : ")
GLCM_param = [int(x.strip()) for x in GLCM_param.split(",")]
location = config['Astropark'][sky_cam]['parameters']['location']
time_zone = config['Astropark'][sky_cam]['parameters']['timezone']
start_date = config['Astropark'][sky_cam]['parameters']['start_date']
mask_path = config['Astropark'][sky_cam]['parameters']['mask']
output_directory = config['Astropark'][sky_cam]['paths']['output_directory']
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
mask = preprocessData().crop_center(mask,crop_size=570)
properties = ['contrast',
            'dissimilarity',
            'homogeneity',
            'energy',
            'correlation',
            'ASM'] 
day = 1
folders = os.listdir(path)
for folder in folders:
    folder = os.path.join(path,folder)
    images,filename = preprocessData().load_images_and_preprocess(folder,mask=mask,apply_crop_sun=True)
    sunrise,sunset = SunPosition().SunriseSunset(filename=filename[50],location=location,Time_zone=time_zone,start_date=start_date,include_end_date=True)
    _,value,statisical = thresholding().RBratio(input=images,filename=filename,factor=[1.17,0.88],sunrise=sunrise,sunset=sunset,Time_zone=time_zone)
    gray = [cv2.cvtColor(i,cv2.COLOR_RGB2GRAY) for i in images]
    glcm = preprocessData().computeGlcm(image=gray,distance=[int(GLCM_param[0])],angle=[GLCM_param[1]])
    df = preprocessData().getDataframe(properties,glcm,index=filename,intensity=value,statistical=statisical)
    df['label'] = os.path.basename(folder)
    output_filename = f'GLCM_ALL_sky_{day}_dis{GLCM_param[0]}_ang{GLCM_param[1]}.csv'
    output_path = os.path.join(output_directory, output_filename)
    print(f"---File {day} write complete---")
    day = day+1
    df.to_csv(output_path)
