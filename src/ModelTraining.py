import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from .TotalCalculation import *
import numpy as np
__all__ = ['TimeCal','dataFilter']

class TimeCal:
    def __init__(self):
        pass

    def TimeFromIndex(self,dataframe,TimeZone:int):
        """
        Calculate time from index of dataframe and create a new column

        Parameters:
        dataframe (Dataframe) : Dataframe that have image filename as index
        TimeZone (int) : offset time from UTC ex. +7 for Thailand,-4 for Chile

        Return:
        dataframe : Dataframe with additional Time in standard time unit

        """
        date = []
        for i in dataframe.index.tolist():
            date.append(timeConvertion.ticks_to_datetime(i,time_zone=TimeZone))
        dataframe['Time'] = date
        dataframe['Time (decimal)'] = dataframe['Time'].dt.hour+dataframe['Time'].dt.minute/60
        return dataframe
    
class dataFilter:
    def __init__(self):
        pass

    def dayTime(self,dataframe,location,start_date:str):
        end_date = str(dataframe['Time'].tolist()[1])
        days = timeConvertion.time_duration(start_date,end_date,include_end_date=True).days
        LSTM = SunPosition.LSTM(time_zone_offset=7)
        EoT = SunPosition.calculate_EoT(day=days)
        TC = SunPosition.TimeCorrectionFactor(Longitude=location[1],LSTM=LSTM,EoT=EoT)
        dec = SunPosition.declination(day=days)
        suntime = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)
        return suntime
    
    def dayTimeFilter(self,dataframe,suntime:list):
        dataframe = dataframe[(dataframe['Time (decimal)'] > suntime[0]) & (dataframe['Time (decimal)'] < suntime[1])]
        return dataframe
    
    def FeatureSelection(self,dataframe,drop_columns:list):
        select = dataframe.drop(columns=drop_columns)
        dataset = list(select.itertuples(index=False,name=None))
        dataset = np.array(dataset)
        return dataset
#Under development