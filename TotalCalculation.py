import time
from datetime import datetime as dt
import datetime
import numpy as np
import cv2

class SunPosition:
    def calculate_EoT(day):
        B = np.deg2rad(360/365*(day - 81))
        return 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    def LSTM(time_zone_offset:int):
        return time_zone_offset*15
    def TimeCorrectionFactor(Longitude,LSTM,EoT):
        return 4*(Longitude-LSTM)+EoT
    def LocalSolarTime(Local_time,TC):
        return Local_time+(TC/60)
    def HourAngle(LST):
        return 15*(LST-12)
    def declination(day):
        return 23.45*np.sin(np.deg2rad(360/365*(day-81)))
    def ElevationandAzimuth(latitude,dec,HRA):
        latitude = np.deg2rad(latitude)
        dec = np.deg2rad(dec)
        HRA = np.deg2rad(HRA)
        elevation = np.arcsin(np.sin(dec)*np.sin(latitude)+
                            np.cos(dec)*np.cos(latitude)*np.cos(HRA))
        azimuth = np.arccos((np.sin(dec)*np.cos(latitude)-np.cos(dec)*np.sin(latitude)*np.cos(HRA))/np.cos(elevation))
        if HRA > 0:
            azimuth = 360-np.degrees(azimuth)
        if HRA < 0:
            azimuth = np.degrees(azimuth)
        return np.degrees(elevation), azimuth
    def AzAtlCalc(date, local_time, longitude, time_zone_offset, latitude):
        EoT = SunPosition.calculate_EoT(date)
        LSTM = SunPosition.LSTM(time_zone_offset)
        TC = SunPosition.TimeCorrectionFactor(longitude, LSTM, EoT)
        LST = SunPosition.LocalSolarTime(local_time, TC)
        HRA = SunPosition.HourAngle(LST)
        dec = SunPosition.declination(date)
        return SunPosition.ElevationandAzimuth(latitude, dec, HRA)
    def DaytimeInfo(latitude,declination,TC):
        latitude = np.deg2rad(latitude)
        declination = np.deg2rad(declination)
        argument = np.tan(latitude) * np.tan(declination)
        sunrise = 12 - (1 / 15) * np.degrees(np.arccos(-argument)) - (TC / 60)
        sunset = 12 + (1 / 15) * np.degrees(np.arccos(-argument)) - (TC / 60)
        return sunrise,sunset
    def SunriseSunset(self,filename:int,start_date,include_end_date:bool):
        start_date = start_date
        end_date = str(timeConvertion().ticks_to_datetime(ticks=filename,time_zone=7).date())
        location = [18.849417,98.9538]
        days = timeConvertion().time_duration(start_date,end_date,include_end_date=include_end_date).days
        LSTM = SunPosition.LSTM(time_zone_offset=7)
        EoT = SunPosition.calculate_EoT(day=days)
        TC = SunPosition.TimeCorrectionFactor(Longitude=location[1],LSTM=LSTM,EoT=EoT)
        dec = SunPosition.declination(day=days)
        sunrise,sunset = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)
        return sunrise,sunset
    
class timeConvertion:
    def __init__(self):
        pass
    def ticks_to_datetime(self,ticks,time_zone:int):
        epoch = dt(1, 1, 1)
        delta = datetime.timedelta(microseconds=ticks/10) 
        result_dt = epoch + delta
        result_dt += datetime.timedelta(hours=time_zone)
        return result_dt
    def time_duration(self,start_date,end_date,include_end_date:bool):
        start_date = dt.strptime(start_date,'%Y-%m-%d').date()
        end_date = dt.strptime(end_date.split()[0],'%Y-%m-%d').date()
        duration = end_date-start_date
        if include_end_date:
            duration += datetime.timedelta(days=1)
        return duration
    def decimal_to_UTC(self,decimal_time, base_date=None):
        hours = int(decimal_time)
        minutes = int((decimal_time - hours) * 60)
        seconds = int(((decimal_time - hours) * 60 - minutes) * 60)
        time_delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return time_delta
    def datetime_to_decimal(self,time):
        return time.hour+time.minute/60+time.second/3600