from code import interact
from re import S
import statistics
import numpy as np
import cv2
import os
from IPython.display import Image
from matplotlib import pyplot as plt
import shutil
from skimage.feature import graycomatrix,graycoprops
import pandas as pd
from .TotalCalculation import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

class preprocessData:
    def __init__(self):
        pass
    def ScaledPCA(self,scaler_path,PCA_path,dataframe,drop_column):
        if drop_column:
            X = dataframe.drop(columns=['homogeneity','skewness','Red channel','different(R-B)'])
        else:
            X = dataframe
        with open(PCA_path,'rb') as PCA_file:
            pca = pickle.load(PCA_file)
        with open(scaler_path,'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        #scaler.partial_fit(X)
        scaled = scaler.transform(X)
        principal = pca.transform(scaled)
        return principal
    def cropSun(self,img):
        """
        Mask sun and equivalent brightness out 

        Parameters:
        img (array) : Image to cut sun out

        return:
        img : Image with the sun crolp out
        """
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
        """
        Calculate skewness of all data 

        Parameters:
        data (list) : Input dat for calculation

        return:
        skewness : Skewness value of the data 
        """
        n = len(data)
        mean_blue = np.mean(data)
        std_dev_blue = np.std(data)
        skewness = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean_blue) / std_dev_blue))
        return skewness
    def crop_center(self,img, crop_size:int):
        h, w = img.shape[:2]
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        end_x = start_x + crop_size
        end_y = start_y + crop_size
        cropped_img = img[start_y:end_y, start_x:end_x]
        return cropped_img
    def load_images_and_preprocess(self, path: str, mask, apply_crop_sun: bool):
        """
        Load images from the specified path, apply optional sun cropping, and mask each image.

        Parameters:
        path (str): Folder path for reading the files.
        mask (array): Mask to apply for removing obstacles.
        apply_crop_sun (bool): If True, remove the sun from the images; otherwise, keep the sun.

        Returns:
        masked (list): List of masked images from the folder.
        name (list): List of image filenames without the extension.
        """
        masked = []
        name = []
        mask_h, mask_w = mask.shape[:2]

        # Precompute resized masks for all unique image sizes encountered
        resize_cache = {}

        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if apply_crop_sun:
                img = self.cropSun(img)

            img_h, img_w = img.shape[:2]

            # Check if the resized mask is already cached; if not, resize and cache it
            if (img_w, img_h) not in resize_cache:
                if img_w == mask_w and img_h == mask_h:
                    resize_cache[(img_w, img_h)] = mask
                else:
                    resize_cache[(img_w, img_h)] = cv2.resize(mask, (img_w, img_h))

            # Apply mask with NumPy instead of OpenCV for better performance
            resized_mask = resize_cache[(img_w, img_h)]
            masked_img = cv2.bitwise_and(img,img,mask=resized_mask)

            masked.append(masked_img)
            name.append(os.path.splitext(filename)[0])  # Store filename without extension

        return masked, name
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
        """
        Create dataframe from input data

        Parameters:
        Property (List) : List of data to create dataframe
        Grey_level (4-D array) : Fix dimension array that have GLCM value calculated from image
        intensity (List) : List of average intensity of each image
        statistical (List) : List of statistical values from each image (See RBratio)

        return:
        dataframe : dataframe of all data with length of the images 
        """
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
        """
        Compute GLCM value of an image

        Parameters:
        image (List) : List of images to calculate GlCM individually
        distance (int) : Distance value for GLCM calculation (eg. 1,2,3,...,n)
        angle (int) : Angle value for GLCM calculation. If angle = 0, GLCM calculation will be horizontal. If angle = 90, GLCM calculation will be perpendicular 

        Returns:
        GLCM : 4 Dimensional array that has GLCM value
        """
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
        """
        Compute GLCM value of an image

        Parameters:
        image (List) : List of images to calculate GlCM individually
        distance (int) : Distance value for GLCM calculation (eg. 1,2,3,...,n)
        angle (int) : Angle value for GLCM calculation. If angle = 0, GLCM calculation will be horizontal. If angle = 90, GLCM calculation will be perpendicular 

        Returns:
        GLCM : 4 Dimensional array that has GLCM value
        """
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
    def RBratio(self,input,filename,factor,Time_zone,sunrise,sunset):
        """
        Compute GLCM value of an image

        Parameters:
        input (List) : Preprocess images
        filename (List) : Filename of each image
        Time_zone (int) : Time zone offset from UTC ex. +7 for Thailand,-4 for Chile 
        Sunrise (float) : Time of sunrise from calculation
        Sunset (float) : Time of sunset from calculation

        return:
        final (List) : List of cloud image that cloud and sky was seperate
        value (List) : List of intensity value of each image
        statistical (List) : This block processes each image, extracts color channel statistics (mean, intensity, skewness, std), 
                             applies a threshold on the ratio of R to B, masks the image, and appends grayscale results 
                             and statistical features (R/B channel difference, etc.) to lists for further analysis.
        """
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
        #day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
        #night_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
        #day_input = [input[i] for i in day_indices]
        #night_input = [input[i] for i in night_indices]
        for i,dec in enumerate(decimal):
            if filtering(dec):
                R,_,B = cv2.split(input[i])
                B = B+e
                chan_b.append(np.mean(B))
                chan_r.append(np.mean(R))
                intensity = np.mean(B)
                ratio = np.log1p(R / (B + 1e-5)) * factor[0]
                ratio = cv2.convertScaleAbs(ratio)
                _, final_mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                masked = cv2.bitwise_and(input[i],input[i],mask=final_mask)
                masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
                final.append(masked_gray)
                value.append(intensity)
                skewness.append(preprocessData().calculate_skewness(B))
                std.append(np.std(B))
                diff.append(np.mean(R-B))
            else:
                R,_,B = cv2.split(input[i])
                B = B+e
                chan_b.append(np.mean(B))
                chan_r.append(np.mean(R))
                intensity = np.mean(B)
                ratio = np.log1p(R / (B + 1e-5)) * factor[1]
                ratio = cv2.convertScaleAbs(ratio)
                _, final_mask = cv2.threshold(ratio, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                masked = cv2.bitwise_and(input[i],input[i],mask=final_mask)
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