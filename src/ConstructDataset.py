import pandas as pd
import numpy as np
import os 
from .preprocessing import thresholding,preprocessData
from .TotalCalculation import timeConvertion,SunPosition
from .ClassPrediction import prediction
import shutil,os

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


