import os 
import pandas as pd
import numpy as np
import cv2
from preprocessing import *

path = input("Enter folder path : ")
mask_path = r'mask_122023.png'
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
    images,filename = load_images_and_preprocess(folder,mask=mask,apply_crop_sun=True)
    masked,value,RB = RBratio(input=images)
    grad = Edging(input=masked,ker_size=7,cliplimit=40,gridsize=14,bias=50)
    glcm = computeGlcm(image=grad,distance=[3],angle=[45])
    df = getDataframe(properties,glcm,index=filename,intensity=value,RB=RB)
    output_filename = f'GLCM_SobelFeature_ALL_sky_{day}dis3_ang45_test_.csv'
    output_path = os.path.join(r'C:\Users\ASUS\Documents\NARIT_internship_data\CSV_dataset_sobel\Dataset_12_2023', output_filename)
    print(f"---File {day} write complete---")
    day = day+1
    df.to_csv(output_path)