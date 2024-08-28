import numpy as np
import cv2
import os
from IPython.display import Image
from matplotlib import pyplot as plt
import shutil
from skimage.feature import graycomatrix,graycoprops
import pandas as pd
from TotalCalculation import *
class preprocessData:
    def __init__(self):
        pass
    def cropSun(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    
        _, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
        try:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)    
            mask = np.zeros_like(gray)
            #cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            area = cv2.countNonZero(mask)
            radius = int(np.sqrt(area / np.pi))
            #img = cv2.circle(img.copy(), (cX, cY), radius, (0, 0, 0), -1)
            img = cv2.drawContours(img.copy(), [largest_contour], -1, 0, thickness=cv2.FILLED)
        except:
            pass
        return img
    
    def crop_center(self,img, crop_size=570):
        h, w = img.shape[:2]
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        cropped_img = img[start_y:end_y, start_x:end_x]
        return cropped_img

    def Threshold(self,image):
        inten = []
        for i in image:
            _,_,B = cv2.split(i)
            inten.append(np.mean(B))
        return np.min(inten[90:400])
    def applySunDelete(self,img):
        return cv2.bitwise_and(img,img,mask=cv2.inRange(img,np.array([0,0,0]),np.array([254,253,255])))
    def load_images_and_preprocess(self,path:str,mask,apply_crop_sun:bool):
        images = []
        name = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            img = img[(int(img.shape[0]/2)-270):int((img.shape[0]/2)+270),int((img.shape[1]/2)-290):int((img.shape[1]/2)+280)]
            img = cv2.bitwise_and(img,img,mask=mask)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if apply_crop_sun:
                img = self.cropSun(img)
            images.append(img)
            name.append(int(os.path.splitext(os.path.basename(filename))[0]))
        return images,name

    def load_single_image(self,path:str,mask:str,apply_crop_sun:bool,crop_size:int):
        images = []
        name = []
        img = cv2.imread(os.path.join(path))
        img = self.crop_center(img,crop_size=crop_size)
        img = cv2.bitwise_and(img,img,mask=mask)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if apply_crop_sun:
            img = self.cropSun(img)
        images.append(img)
        name.append(int(os.path.splitext(os.path.basename(path))[0]))
        return images,name
    def Edging(self,input:list,ker_size:int,cliplimit:int,gridsize:int,bias:int):
        grad = []
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(gridsize, gridsize))
        for i in input:
            i = clahe.apply(i)
            gre = ((cv2.Sobel(i, cv2.CV_64F, 0, 1, ksize=ker_size)+cv2.Sobel(i, cv2.CV_64F, 1, 0, ksize=ker_size))/1000)+bias
            _,thresh = cv2.threshold(gre,0,255,cv2.THRESH_TOZERO)
            grad.append(cv2.convertScaleAbs(thresh))
        return grad
    def getDataframe(self,property:list,gray_level,index:list,intensity,RB):
        dataset = {
            prop : [] for prop in property
        }
        for i in gray_level:
            for prop in property:
                dataset[prop].append(graycoprops(i, prop).flatten()[0])
        dataframe = pd.DataFrame(data=dataset,index=index)
        if intensity:
            dataframe['intensity'] = [np.mean(i) for i in intensity]
            dataframe['Red channel'] = [np.mean(i) for i in RB[0]]
            dataframe['Blue channel'] = [np.mean(i) for i in RB[1]]
        return dataframe

    def computeGlcm(self,image:list,distance,angle):
        glcm = []
        for i in image:
            gray = graycomatrix(i,distance,angle)
            glcm.append(gray)
        return glcm
    def showCloudRatio(self,images,mask,number):
        image = images
        return cv2.countNonZero(image[number])/cv2.countNonZero(mask)
    
class image:
    def getFilename(path):
        img = []
        for filename in os.listdir(path):
            img.append(os.path.join(path,filename))
        return img
    def extract_filename(filename:list,sunrise:float,sunset:float):
        extract_name = lambda x : int(os.path.splitext(os.path.basename(x))[0])
        filtering = lambda x : (x > sunrise) & (x < sunset)

        filetime = [extract_name(i) for i in filename]
        decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t,time_zone=7)) for t in filetime]
        day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
        night_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
        Day_filename = [filename[i] for i in day_indices]
        night_filename = [filename[i] for i in night_indices]
        return Day_filename,night_filename

class thresholding:
    def __init__(self):
        pass
#    def RBratioNight(self,input):
#        final = []
#        value= []
#        chan_b = []
#        chan_r = []
#        e = 1e-11
#        threshold = preprocessData.Threshold(input)
#        for i in input:
#            R,_,B = cv2.split(i)
#            B = B+e
#            chan_b.append(np.mean(B))
#            chan_r.append(np.mean(R))
#            if intensity < threshold:
#                ratio = cv2.convertScaleAbs(ratio)
#                final_mask = cv2.threshold(ratio, intensity/8, 255, cv2.THRESH_BINARY)[1]
#               masked = cv2.bitwise_and(i,i,mask=final_mask)
#               masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)*2
#        chan_r = np.array(chan_r).reshape(-1,1)
#        chan_b = np.array(chan_b).reshape(-1,1)
#        return final,value,RB.T
    def RBratio(self,input):
        final = []
        value= []
        chan_b = []
        chan_r = []
        e = 1e-11
        for i in input:
            R,_,B = cv2.split(i)
            B = B+e
            chan_b.append(np.mean(B))
            chan_r.append(np.mean(R))
            intensity = np.mean(B)
            ratio = (R/B)*intensity/9
            ratio = cv2.convertScaleAbs(ratio)
            final_mask = cv2.threshold(ratio, intensity/20, 255, cv2.THRESH_BINARY)[1]

            masked = cv2.bitwise_and(i,i,mask=final_mask)
            masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
            final.append(masked_gray)
            value.append(intensity)
        chan_r = np.array(chan_r).reshape(-1,1)
        chan_b = np.array(chan_b).reshape(-1,1)
        RB = np.concatenate((chan_r,chan_b),axis=1)
        return final,value,RB.T