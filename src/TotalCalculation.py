import time
from datetime import datetime as dt
import datetime
import numpy as np
import cv2

"""
class CloudCalculation 

Calculate cloud ratio

showCloudRatio(images,mask,number) : Calculate cloud percentage from image compares to the image mask
    Parameters:
        images (List) : Preprocessed images in list
        mask (array) : Mask image for obstacle elimination
        number (int) : Number of image to show
    Return:
        float : Cloud ratio in the sky from 0-1
"""
class CloudCalculation:
    def __init__(self):
        pass    
    def showCloudRatio(self,images,mask,number):
        image = images
        return cv2.countNonZero(image[number])/cv2.countNonZero(mask)

class SunPosition:
    """
    class SunPosition

    Calculate sun position and daytime

    Methods:
        - calculate_EoT(day): Computes the Equation of Time (EoT) based on the day of the year.
        - LSTM(time_zone_offset): Calculates the Local Standard Time Meridian (LSTM) using time zone offset.
        - TimeCorrectionFactor(Longitude, LSTM, EoT): Computes time correction factor based on location.
        - LocalSolarTime(Local_time, TC): Calculates Local Solar Time using the local time and time correction.
        - HourAngle(LST): Calculates the Hour Angle from Local Solar Time.
        - declination(day): Computes the solar declination angle for the given day of the year.
        - ElevationandAzimuth(latitude, dec, HRA): Returns the solar elevation and azimuth angles.
        - AzAtlCalc(date, local_time, longitude, time_zone_offset, latitude): Combines several methods to compute the sun's position (elevation and azimuth).
        - DaytimeInfo(latitude, declination, TC): Calculates the sunrise and sunset times based on the solar declination.
        - SunriseSunset(location, filename, start_date, Time_zone, include_end_date): Returns sunrise and sunset times for a given location and time range.
    """
    def calculate_EoT(day):
        """
        Calculate the Equation of Time (EoT).

        Parameters:
            day (int): Day of the year (1 to 365).

        Returns:
            float: The Equation of Time in minutes.
        """
        B = np.deg2rad(360/365*(day - 81))
        return 9.87*np.sin(2*B)-7.53*np.cos(B)-1.5*np.sin(B)
    def LSTM(time_zone_offset:int):
        """
        Calculate the Local Standard Time Meridian (LSTM).

        Parameters:
            time_zone_offset (int): Time zone offset from UTC.

        Returns:
            int: LSTM in degrees.
        """
        return time_zone_offset*15
    def TimeCorrectionFactor(Longitude,LSTM,EoT):
        """
        Calculate the Time Correction Factor (TC).

        Parameters:
            Longitude (float): Longitude of the location.
            LSTM (float): Local Standard Time Meridian (in degrees).
            EoT (float): Equation of Time (in minutes).

        Returns:
            float: Time Correction Factor in minutes.
        """
        return 4*(Longitude-LSTM)+EoT
    def LocalSolarTime(Local_time,TC):
        """
        Calculate Local Solar Time (LST).

        Parameters:
            Local_time (float): Local time in hours.
            TC (float): Time Correction Factor (in minutes).

        Returns:
            float: Local Solar Time in hours.
        """
        return Local_time+(TC/60)
    def HourAngle(LST):
        """
        Calculate the Hour Angle (HRA).

        Parameters:
            LST (float): Local Solar Time in hours.

        Returns:
            float: Hour Angle in degrees.
        """
        return 15*(LST-12)
    def declination(day):
        """
        Calculate the solar declination angle.

        Parameters:
            day (int): Day of the year (1 to 365).

        Returns:
            float: Declination angle in degrees.
        """
        return 23.45*np.sin(np.deg2rad(360/365*(day-81)))
    def ElevationandAzimuth(latitude,dec,HRA):
        """
        Calculate the solar elevation and azimuth angles.

        Parameters:
            latitude (float): Latitude of the location in degrees.
            dec (float): Solar declination angle in degrees.
            HRA (float): Hour Angle in degrees.

        Returns:
            tuple: Solar elevation (in degrees) and azimuth (in degrees).
        """
        latitude = np.deg2rad(latitude)
        dec = np.deg2rad(dec)
        HRA = np.deg2rad(HRA)
        elevation = np.arcsin(np.sin(dec)*np.sin(latitude)+
                            np.cos(dec)*np.cos(latitude)*np.cos(HRA))
        azimuth = np.arccos((np.sin(dec)*np.cos(latitude)-np.cos(dec)*
                             np.sin(latitude)*np.cos(HRA))/np.cos(elevation))
        if HRA > 0:
            azimuth = 360-np.degrees(azimuth)
        if HRA < 0:
            azimuth = np.degrees(azimuth)
        return np.degrees(elevation), azimuth
    def AzAtlCalc(date, local_time, longitude, time_zone_offset, latitude):
        """
        Calculate the sun's azimuth and elevation based on date, time, and location.

        Parameters:
            date (int): Day of the year (1 to 365).
            local_time (float): Local time in hours.
            longitude (float): Longitude of the location in degrees.
            time_zone_offset (int): Time zone offset from UTC.
            latitude (float): Latitude of the location in degrees.

        Returns:
            tuple: Solar elevation (in degrees) and azimuth (in degrees).
        """
        EoT = SunPosition.calculate_EoT(date)
        LSTM = SunPosition.LSTM(time_zone_offset)
        TC = SunPosition.TimeCorrectionFactor(longitude, LSTM, EoT)
        LST = SunPosition.LocalSolarTime(local_time, TC)
        HRA = SunPosition.HourAngle(LST)
        dec = SunPosition.declination(date)
        return SunPosition.ElevationandAzimuth(latitude, dec, HRA)
    def DaytimeInfo(latitude,declination,TC):
        """
        Calculate sunrise and sunset times.

        Parameters:
            latitude (float): Latitude of the location in degrees.
            declination (float): Solar declination angle in degrees.
            TC (float): Time Correction Factor (in minutes).

        Returns:
            tuple: Sunrise time (in hours) and sunset time (in hours).
        """
        latitude = np.deg2rad(latitude)
        declination = np.deg2rad(declination)
        argument = np.tan(latitude) * np.tan(declination)
        sunrise = 12 - (1 / 15) * np.degrees(np.arccos(-argument)) - (TC / 60)
        sunset = 12 + (1 / 15) * np.degrees(np.arccos(-argument)) - (TC / 60)
        return sunrise,sunset
    def SunriseSunset(self,location:list,filename:int,start_date,Time_zone:int,include_end_date:bool):
        """
        Calculate sunrise and sunset times for a given location and date range.

        Parameters:
            location (list): List containing latitude and longitude of the location.
            filename (int): Filename as a timestamp (used to determine end date).
            start_date (str): Start date in 'YYYY-MM-DD' format.
            Time_zone (int): Time zone offset from UTC.
            include_end_date (bool): Whether to include the end date in the calculation.

        Returns:
            tuple: Sunrise time (in hours) and sunset time (in hours) for the specified date range.
        """
        start_date = start_date
        end_date = str(timeConvertion().ticks_to_datetime(ticks=filename,time_zone=Time_zone).date())
        location = location
        days = timeConvertion().time_duration(start_date,end_date,include_end_date=include_end_date).days
        LSTM = SunPosition.LSTM(time_zone_offset=Time_zone)
        EoT = SunPosition.calculate_EoT(day=days)
        TC = SunPosition.TimeCorrectionFactor(Longitude=location[1],LSTM=LSTM,EoT=EoT)
        dec = SunPosition.declination(day=days)
        sunrise,sunset = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)
        return sunrise,sunset

class timeConvertion:
    """
    A class for converting time between different formats such as ticks to datetime, calculating 
    time duration, converting decimal time to UTC, and converting datetime to decimal time.

    Methods:
        - ticks_to_datetime(ticks, time_zone): Converts ticks since epoch (in microseconds) to a datetime object, adjusted for the time zone.
        - time_duration(start_date, end_date, include_end_date): Calculates the duration between two dates, with the option to include the end date.
        - decimal_to_UTC(decimal_time, base_date): Converts decimal time to a time delta.
        - datetime_to_decimal(time): Converts a datetime object to decimal hours.
    """
    def __init__(self):
        pass
    def ticks_to_datetime(self,ticks,time_zone:int):
        """
        Convert ticks to a datetime object, adjusted for time zone.

        Parameters:
            ticks (int): The number of ticks (in microseconds) since epoch (January 1, 1 A.D.).
            time_zone (int): The time zone offset from UTC (in hours).

        Returns:
            datetime: The resulting datetime object after converting ticks and adjusting for the time zone.
        """
        epoch = dt(1, 1, 1)
        delta = datetime.timedelta(microseconds=ticks/10) 
        result_dt = epoch + delta
        result_dt += datetime.timedelta(hours=time_zone)
        return result_dt
    def time_duration(self,start_date,end_date,include_end_date:bool):
        """
        Calculate the duration between two dates, optionally including the end date.

        Parameters:
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            include_end_date (bool): Whether to include the end date in the calculation.

        Returns:
            timedelta: Duration between the two dates, including end date if specified.
        """
        start_date = dt.strptime(start_date,'%Y-%m-%d').date()
        end_date = dt.strptime(end_date.split()[0],'%Y-%m-%d').date()
        duration = end_date-start_date
        if include_end_date:
            duration += datetime.timedelta(days=1)
        return duration
    def decimal_to_UTC(self,decimal_time, base_date=None):
        """
        Convert decimal time to a time delta.

        Parameters:
            decimal_time (float): Decimal time in hours (e.g., 14.5 represents 14:30).
            base_date (datetime, optional): Base date to add the time to, defaults to None.

        Returns:
            timedelta: The corresponding time delta (hours, minutes, and seconds).
        """
        hours = int(decimal_time)
        minutes = int((decimal_time - hours) * 60)
        seconds = int(((decimal_time - hours) * 60 - minutes) * 60)
        time_delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return time_delta
    def datetime_to_decimal(self,time):
        """
        Convert a datetime object to decimal hours.

        Parameters:
            time (datetime): A datetime object representing the time to convert.

        Returns:
            float: Time in decimal hours (e.g., 14:30 becomes 14.5).
        """
        return time.hour+time.minute/60+time.second/3600