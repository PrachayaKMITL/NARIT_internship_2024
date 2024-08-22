import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from TotalCalculation import *

class TimeCal:
    def __init__(self):
        pass

    def TimeFromIndex(self,dataframe,TimeZone:int):
        date = []
        for i in dataframe.index.tolist():
            date.append(timeConvertion.ticks_to_datetime(i,time_zone=TimeZone))
        dataframe['Time'] = date
        dataframe['Time (decimal)'] = dataframe['Time'].dt.hour+dataframe['Time'].dt.minute/60
        return dataframe
    
class dataFilter:
    def __init__(self):
        pass

    def dayTime(dataframe):
        start_date = '2024-01-01'
        end_date = str(dataframe['Time'].tolist()[1])
        location = [18.849417,98.9538]
        days = timeConvertion.time_duration(start_date,end_date,include_end_date=True).days
        LSTM = SunPosition.LSTM(time_zone_offset=7)
        EoT = SunPosition.calculate_EoT(day=days)
        TC = SunPosition.TimeCorrectionFactor(Longitude=location[1],LSTM=LSTM,EoT=EoT)
        dec = SunPosition.declination(day=days)
        suntime = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)
        return suntime
    
    def dayTimeFilter(dataframe,suntime:list):
        dataframe = dataframe[(dataframe['Time (decimal)'] > suntime[0]) & (dataframe['Time (decimal)'] < suntime[1])]
        return dataframe
