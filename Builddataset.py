import os 
import json
import pandas as pd
import numpy as np
import cv2
from src.preprocessing import *
from src.ConstructDataset import Builddataset
sky_cam = str(input("Select images to train(Astropark : (December,April,June)) : "))
with open("Configuration\\Dataset_configuration.json", 'r') as config_file:
    config = json.load(config_file)
path = input("Enter folder path : ")
GLCM_param = list(input("Enter distance and angle in list : "))
location = config['Astropark'][sky_cam]['parameters']['location']
time_zone = config['Astropark'][sky_cam]['parameters']['timezone']
start_date = config['Astropark'][sky_cam]['parameters']['start_date']
mask_path = config['Astropark'][sky_cam]['parameters']['mask']
output_directory = config['Astropark'][sky_cam]['paths']['output_directory']
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
for folder in folders:
    folder = os.path.join(path,folder)
    images,filename = preprocessData().load_images_and_preprocess(folder,mask=mask,apply_crop_sun=True)
    #masked,value,statisical = thresholding().RBratio(input=images,filename=filename,sunrise=sunrise,sunset=sunset,Time_zone=time_zone)
    gray_image = [cv2.cvtColor(i,cv2.COLOR_RGB2GRAY) for i in images]
    intensity,statistic = Builddataset().Statistical(input=images)
    glcm = preprocessData().computeGlcm(image=gray_image,distance=[int(GLCM_param[1])],angle=[GLCM_param[3]])
    df = preprocessData().getDataframe(properties,glcm,index=filename,intensity=intensity,statistical=statistic)
    output_filename = f'GLCM_ALL_sky_{day}_dis{GLCM_param[1]}_ang{GLCM_param[3]}.csv'
    output_path = os.path.join(output_directory, output_filename)
    print(f"---File {day} write complete---")
    day = day+1
    df.to_csv(output_path)
