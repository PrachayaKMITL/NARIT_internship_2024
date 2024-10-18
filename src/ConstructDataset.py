import pandas as pd
import numpy as np
import os 
from .preprocessing import thresholding,preprocessData
from .TotalCalculation import timeConvertion,SunPosition
from .ClassPrediction import prediction
import shutil,os
import cv2

__all__ = ['Builddataset']

class Builddataset:
    def __init__(self):
        pass        
    def concateDataset(self,folder_name):
        """
        Concatenate dataset generated from Builddataset.py

        Parameters:
        folder_name (str) : Path of the folder that dataset is located

        Returns:
        dataframe : Concatenated dataframe of every dataframe in the dataset
        """
        data_list = []
        for filename in os.listdir(folder_name):
            name = os.path.join(folder_name,filename)
            data_list.append(pd.read_csv(name,index_col='Unnamed: 0'))
        return pd.concat(data_list)
    def DayNightSplit(self,suntime,dataframe,Mode:str):
        """
        Seperate between day and night for prediction and training
        
        paramters:
        suntime (List) : List of sunrise time and sunset time respectively
        dataframe : Dataframe to applied split on
        Mode (str) : Selection between day and night

        Returns:
        Dataframe : Dataframe that satisfied condition
        """
        if Mode == 'day':
            dataframe = dataframe[(dataframe['Time (decimal)'] > suntime[0]) & (dataframe['Time (decimal)'] < suntime[1])]
        if Mode == 'night':
            dataframe = dataframe[(dataframe['Time (decimal)'] < suntime[0]) | (dataframe['Time (decimal)'] > suntime[1])]
        return dataframe
    def Statistical(self,input:list):
        """
        Compute statical value of each image

        Parameters:
        input (List) : Preprocess images

        return:
        value (List) : List of intensity value of each image
        statistical (List) : This block processes each image, extracts color channel statistics (mean, intensity, skewness, std), 
                             applies a threshold on the ratio of R to B, masks the image, and appends grayscale results 
                             and statistical features (R/B channel difference, etc.) to lists for further analysis.
        """
        intensity = []
        chan_b = []
        chan_r = []
        skewness = []
        std = []
        diff = []
        for i in input:
            R,_,B = cv2.split(i)
            B = B + 1e-11
            intensity.append(np.mean(B))
            skewness.append(preprocessData().calculate_skewness(B))
            std.append(np.std(B))
            diff.append(np.mean(R-B))
            chan_b.append(np.mean(B))
            chan_r.append(np.mean(R))
        chan_r = np.array(chan_r).reshape(-1,1)
        chan_b = np.array(chan_b).reshape(-1,1)
        skewness = np.array(skewness).reshape(-1,1)
        std = np.array(std).reshape(-1,1)
        diff = np.array(diff).reshape(-1,1)
        statistical = np.concatenate((skewness,std,diff,chan_r,chan_b),axis=1)

        return intensity,statistical.T

    def Statistical_test(self, input: list):
        """
        Compute statistical values of each image.

        Parameters:
        input (List) : Preprocessed images

        Returns:
        intensity (List) : List of intensity values of each image
        statistical (Array) : Array containing mean, skewness, std, R/B difference, and means of R and B channels.
        """
        num_images = len(input)
        
        # Preallocate arrays
        intensity = np.zeros(num_images)
        chan_r = np.zeros(num_images)
        chan_b = np.zeros(num_images)
        skewness = np.zeros(num_images)
        std = np.zeros(num_images)
        diff = np.zeros(num_images)

        for idx, img in enumerate(input):
            R, _, B = cv2.split(img)

            intensity[idx] = np.mean(img)
            skewness[idx] = self.calculate_skewness(B)
            std[idx] = np.std(B)
            diff[idx] = np.mean(R - B)
            chan_r[idx] = np.mean(R)
            chan_b[idx] = np.mean(B)

        statistical = np.column_stack((intensity, skewness, std, diff, chan_r, chan_b))

        return intensity.tolist(), statistical


