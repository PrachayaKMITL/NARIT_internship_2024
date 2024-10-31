import os
import json
import pandas as pd
import numpy as np
import cv2
import gc
import logging
from src.preprocessing import preprocessData
from src.ConstructDataset import Builddataset
import time

log_dir = "logs/Dataset_log"
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"dataset_generation_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open("Configuration\\Dataset_configuration.json", 'r') as config_file:
    config = json.load(config_file)
print(json.dumps(config, indent=1))

logging.info("Configuration loaded successfully.")

sky_cam = str(input("Select configuration based on location: "))
path = input("Enter folder path: ")
GLCM_param = input("Enter distance and angle in list: ")
GLCM_param = [int(x.strip()) for x in GLCM_param.split(",")]

logging.info(f"User selected configuration: {sky_cam}")
logging.info(f"Dataset path: {path}")
logging.info(f"GLCM parameters: {GLCM_param}")

location = config[sky_cam]['parameters']['location']
time_zone = config[sky_cam]['parameters']['timezone']
start_date = config[sky_cam]['parameters']['start_date']
mask_path = config[sky_cam]['parameters']['mask']
output_directory = config[sky_cam]['paths']['output_directory']

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
logging.info(f"Mask loaded from: {mask_path}")
properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

output_folder = os.path.join(output_directory, f"GLCM_dis-{GLCM_param[0]}_ang-{GLCM_param[1]}")
os.makedirs(output_folder, exist_ok=True)

logging.info(f"Output directory created: {output_folder}")

folders = os.listdir(path)
for folder in folders:
    class_folder = os.path.join(path, folder)
    logging.info(f"Processing folder: {class_folder}")

    GLCM = []
    Filename = []
    Intensity = []
    skewness = []
    std = []
    diff = []
    chan_r = []
    chan_b = []

    for filename in os.listdir(class_folder):
        img_path = os.path.join(class_folder, filename)

        images, name = preprocessData().load_single_image(path=img_path, mask=mask, apply_crop_sun=True)

        value, statistical = Builddataset().Statistical(input=images)
        gray = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in images]

        glcm = preprocessData().computeGlcm(image=gray, distance=[int(GLCM_param[0])], angle=[GLCM_param[1]])
        GLCM.append(glcm[0])

        Filename.append(name[0])
        Intensity.append(value[0])
        skewness.append(statistical[0][0])
        std.append(statistical[1][0])
        diff.append(statistical[2][0])
        chan_r.append(statistical[3][0])
        chan_b.append(statistical[4][0])

    Statistical = np.array([skewness, std, diff, chan_r, chan_b])
    df = preprocessData().getDataframe(properties, gray_level=GLCM, index=Filename, intensity=Intensity, statistical=Statistical)
    sky_cat = os.path.basename(folder)
    df['label'] = sky_cat
    output_filename = f'GLCM_ALL_sky_{sky_cat}_dis{GLCM_param[0]}_ang{GLCM_param[1]}.csv'
    output_path = os.path.join(output_folder, output_filename)
    df.to_csv(output_path)
    
    logging.info(f"CSV file saved: {output_path}")
    print(f"---{os.path.basename(folder)} file write complete---")
    gc.collect()
    logging.info(f"Memory cleared after processing folder: {folder}")
logging.info("Dataset generation completed.")
print("Dataset generation completed.")
