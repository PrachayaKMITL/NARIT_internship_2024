import os 
import json
import pandas as pd
import numpy as np
import cv2
import gc

from src.preprocessing import *
from src.ConstructDataset import Builddataset

with open("Configuration\\Dataset_configuration.json", 'r') as config_file:
    config = json.load(config_file)
print(json.dumps(config, indent=1))
sky_cam = str(input("Select configuration based on location : "))
path = input("Enter folder path : ")
GLCM_param = input("Enter distance and angle in list : ")
GLCM_param = [int(x.strip()) for x in GLCM_param.split(",")]
location = config[sky_cam]['parameters']['location']
time_zone = config[sky_cam]['parameters']['timezone']
start_date = config[sky_cam]['parameters']['start_date']
mask_path = config[sky_cam]['parameters']['mask']
output_directory = config[sky_cam]['paths']['output_directory']
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
properties = ['contrast',
            'dissimilarity',
            'homogeneity',
            'energy',
            'correlation',
            'ASM'] 
day = 1
folders = os.listdir(path)
output_folder = os.path.join(output_directory,f"GLCM_dis-{GLCM_param[0]}_ang-{GLCM_param[1]}")
os.makedirs(output_folder,exist_ok=True)
for folder in folders:
    folder = os.path.join(path,folder)
    images,filename = preprocessData().load_images_and_preprocess(folder,mask=mask,apply_crop_sun=True)
    images = [cv2.resize(i,(1036,705)) for i in images]
    images = [preprocessData().crop_center(i,crop_size=570) for i in images]
    value,statisical = Builddataset().Statistical(input=images)
    gray = [cv2.cvtColor(i,cv2.COLOR_RGB2GRAY) for i in images]
    del images
    glcm = preprocessData().computeGlcm(image=gray,distance=[int(GLCM_param[0])],angle=[GLCM_param[1]])
    del gray
    df = preprocessData().getDataframe(properties,glcm,index=filename,intensity=value,statistical=statisical)
    del statisical,value
    sky_cat =  os.path.basename(folder)
    df['label'] = sky_cat
    output_filename = f'GLCM_ALL_sky_{sky_cat}_dis{GLCM_param[0]}_ang{GLCM_param[1]}.csv'
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path)
    print(f"---{os.path.basename(folder)} file write complete---")

    del glcm,df
    gc.collect()