from code import interact
import statistics
import numpy as np
import cv2
import os
from IPython.display import Image
from matplotlib import pyplot as plt
import shutil
from skimage.feature import graycomatrix,graycoprops
import pandas as pd
from TotalCalculation import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class preprocessData:
    def __init__(self):
        pass
    def ScaledPCA(self,scaler,dataframe):
        x = dataframe[['contrast','std','dissimilarity','ASM','energy']]
        scaled = scaler.fit_transform(x)
        #pca = PCA(n_components=components,svd_solver="full")
        #principal = pca.fit_transform(scaled)
        return scaled
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
    def calculate_skewness(self,data):
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Using ddof=1 for sample standard deviation
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std_dev) ** 3)
        return skewness
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
    def getDataframe(self,property:list,gray_level,index:list,intensity,statistical):
        dataset = {
            prop : [] for prop in property
        }
        for i in gray_level:
            for prop in property:
                dataset[prop].append(graycoprops(i, prop).flatten()[0])
        dataframe = pd.DataFrame(data=dataset,index=index)
        if intensity is not None:
            dataframe['intensity'] = [np.mean(i) for i in intensity]
        if statistical is not None:
            dataframe['Red channel'] = [np.mean(i) for i in statistical[3]]
            dataframe['Blue channel'] = [np.mean(i) for i in statistical[4]]
            dataframe['skewness'] = statistical[0]
            dataframe['std'] = statistical[1]
            dataframe['different(R-B)'] = statistical[2]
        return dataframe
    def computeGlcmsingle(self,image,distance,angle):
        glcm = graycomatrix(image,distance,angle)
        return glcm
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
    def filename_to_ticks(self,filename:int):
        extract_name = lambda x : int(os.path.splitext(os.path.basename(x))[0])
        filetime = [extract_name(i) for i in filename]
        return filetime
    def extract_filename(filename:list,sunrise:float,sunset:float, time_zone:int):
        extract_name = lambda x : int(os.path.splitext(os.path.basename(x))[0])
        filtering = lambda x : (x > sunrise) & (x < sunset)

        filetime = [extract_name(i) for i in filename]
        decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t,time_zone=time_zone)) for t in filetime]
        day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
        night_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
        Day_filename = [filename[i] for i in day_indices]
        night_filename = [filename[i] for i in night_indices]
        return Day_filename,night_filename

class thresholding:
    def __init__(self):
        pass
    def RBratiosingle(self,input,filename,sunrise,sunset):
        filtering = lambda x : (x > sunrise) & (x < sunset)
        decimal = timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=filename,time_zone=7))
        if filtering(decimal):
            R,_,B = cv2.split(input)
            intensity = np.mean(B)
            ratio = np.log1p(R / (B + 1e-5)) * 1.25
            ratio = cv2.convertScaleAbs(ratio)
            _, final_mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(input,input,mask=final_mask)
            masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
            skewness = preprocessData().calculate_skewness(B)
            std = np.std(B)
            diff = np.mean(R-B)
        if not filtering(decimal):
            R,_,B = cv2.split(input)
            intensity = np.mean(B)
            ratio = np.log1p(R / (B + 1e-5)) * 0.9
            ratio = cv2.convertScaleAbs(ratio)
            _, final_mask = cv2.threshold(ratio, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(input,input,mask=final_mask)
            masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
            skewness = preprocessData().calculate_skewness(B)
            std = np.std(B)
            diff = np.mean(R-B)
        statistical = [skewness,std,diff,[R],[B]]
        return masked_gray,intensity,statistical
    def RBratio(self,input,filename,Time_zone,sunrise,sunset):
        final = []
        value= []
        chan_b = []
        chan_r = []
        skewness = []
        std = []
        diff = []
        e = 1e-11
        filtering = lambda x : (x > sunrise) & (x < sunset)
        decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t,time_zone=Time_zone)) for t in filename]
        day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
        night_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
        day_input = [input[i] for i in day_indices]
        night_input = [input[i] for i in night_indices]
        for i in day_input:
            R,_,B = cv2.split(i)
            B = B+e
            chan_b.append(np.mean(B))
            chan_r.append(np.mean(R))
            intensity = np.mean(B)
            ratio = np.log1p(R / (B + 1e-5)) * 1.2
            ratio = cv2.convertScaleAbs(ratio)
            _, final_mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(i,i,mask=final_mask)
            masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
            final.append(masked_gray)
            value.append(intensity)
            skewness.append(preprocessData().calculate_skewness(B))
            std.append(np.std(B))
            diff.append(np.mean(R-B))
        for i in night_input:
            R,_,B = cv2.split(i)
            B = B+e
            chan_b.append(np.mean(B))
            chan_r.append(np.mean(R))
            intensity = np.mean(B)
            ratio = np.log1p(R / (B + 1e-5)) * 1.2
            ratio = cv2.convertScaleAbs(ratio)
            _, final_mask = cv2.threshold(ratio, 2, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masked = cv2.bitwise_and(i,i,mask=final_mask)
            masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
            final.append(masked_gray)
            value.append(intensity)
            skewness.append(preprocessData().calculate_skewness(B))
            std.append(np.std(B))
            diff.append(np.mean(R-B))
        chan_r = np.array(chan_r).reshape(-1,1)
        chan_b = np.array(chan_b).reshape(-1,1)
        skewness = np.array(skewness).reshape(-1,1)
        std = np.array(std).reshape(-1,1)
        diff = np.array(diff).reshape(-1,1)
        statistical = np.concatenate((skewness,std,diff,chan_r,chan_b),axis=1)
        return final,value,statistical.T