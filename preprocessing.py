import numpy as np
import cv2
import os
from IPython.display import Image
from matplotlib import pyplot as plt
import shutil
from skimage.feature import graycomatrix,graycoprops
import pandas as pd

class Augmentation:
    def __init__(self):
        pass
    
    def readFilenmame(path):
        image_append = []
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path,filename))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image_append.append(img)
        return image_append


    def addNoise(self,img_list,mean,st):
        image_noise = []
        for i in img_list:
            image=i
            mean=mean
            st=st
            gauss = np.random.normal(mean,st,i.shape)
            gauss = gauss.astype('uint8')

            image = cv2.add(image,gauss)
            image_noise.append(image)
        
        return image_noise

    def addBlur(self,img_list,f_size):
        image_blur = []
        for i in img_list:
            image=i
            image_blur.append(cv2.blur(image,(f_size,f_size)))
        for i in img_list:
            image=i
            image_blur.append(cv2.GaussianBlur(image,(f_size,f_size),0))
        for i in img_list:
            image=i
            image_blur.append(cv2.medianBlur(image,f_size))
        
        return image_blur
    
    def merge(self,image_noise,image_blur):
        merge_image = image_noise+image_blur
        return merge_image
    
    def save_images(self,img_list, output_folder, verbose):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        
            for i, img in enumerate(img_list):
                img_name = f"cloud_augment_{i+1}.jpg"  # Adjust the naming convention as needed
                img_path = os.path.join(output_folder, img_name)
                rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, rgb)
                if verbose is True:
                    print("All_img_saved!")
        except:
            print("Error")


class preprocess:
    def __init__(self):
        pass
    
    def Masking(image,mask_path: str):
        img = []
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        for i in image:
            img = cv2.bitwise_and(i,i,mask=mask)
            img.append(img)
        return img
    
    def delete_sun(image,lower: np.array,upper: np.array):
        img = []
        for i in image:
            range = cv2.inRange(i,lower,upper)
            img = cv2.bitwise_and(i,i,mask=range)
        return img

def Threshold(image):
    inten = []
    for i in image:
        _,_,B = cv2.split(i)
        inten.append(np.mean(B))
    return np.min(inten[90:400])
def RBratio(input):
    final = []
    value= []
    chan_b = []
    chan_r = []
    e = 1e-11
    threshold = Threshold(input)
    for i in input:
        R,_,B = cv2.split(i)
        B = B+e
        chan_b.append(np.mean(B))
        chan_r.append(np.mean(R))
        intensity = np.mean(B)
        if intensity < threshold:
            ratio = (R/B)*intensity/8
            ratio = cv2.convertScaleAbs(ratio)
            final_mask = cv2.threshold(ratio, intensity/10, 255, cv2.THRESH_BINARY)[1]
        if intensity >= threshold:
            ratio = (R/B)*intensity/10
            ratio = cv2.convertScaleAbs(ratio)
            final_mask = cv2.threshold(ratio, intensity/20, 255, cv2.THRESH_BINARY)[1]
        '''if intensity < threshold-26:
            ratio = (R/B)*intensity/8
            ratio_1 = cv2.convertScaleAbs(ratio)
            _,final_mask = cv2.threshold(ratio_1, intensity/16, 255, cv2.THRESH_BINARY)'''

        masked = cv2.bitwise_and(i,i,mask=final_mask)
        masked_gray = cv2.cvtColor(masked,cv2.COLOR_RGB2GRAY)
        final.append(masked_gray)
        value.append(intensity)
    chan_r = np.array(chan_r).reshape(-1,1)
    chan_b = np.array(chan_b).reshape(-1,1)
    RB = np.concatenate((chan_r,chan_b),axis=1)
    return final,value,RB.T
def applySunDelete(img):
    return cv2.bitwise_and(img,img,mask=cv2.inRange(img,np.array([0,0,0]),np.array([254,253,255])))
def load_images_and_preprocess(path:str,mask,apply_crop_sun:bool):
    images = []
    name = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        img = img[(int(img.shape[0]/2)-270):int((img.shape[0]/2)+270),int((img.shape[1]/2)-290):int((img.shape[1]/2)+280)]
        img = cv2.bitwise_and(img,img,mask=mask)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if apply_crop_sun:
           img = cropSun(img)
        images.append(img)
        name.append(int(os.path.splitext(os.path.basename(filename))[0]))
    return images,name

def load_single_image(path:str,mask:str,apply_crop_sun:bool,crop_size:int):
    images = []
    name = []
    img = cv2.imread(os.path.join(path))
    img = crop_center(img,crop_size=crop_size)
    img = cv2.bitwise_and(img,img,mask=mask)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if apply_crop_sun:
        img = cropSun(img)
    images.append(img)
    name.append(int(os.path.splitext(os.path.basename(path))[0]))
    return images,name

def getDataframe(property:list,gray_level,index:list,intensity,RB):
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
def crop_center(img, crop_size=570):
    h, w = img.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    end_x = start_x + crop_size
    end_y = start_y + crop_size
    cropped_img = img[start_y:end_y, start_x:end_x]
    return cropped_img

def computeGlcm(image:list,distance,angle):
    glcm = []
    for i in image:
        gray = graycomatrix(i,distance,angle)
        glcm.append(gray)
    return glcm
def showCloudRatio(images,mask,number):
    image = images
    return cv2.countNonZero(image[number])/cv2.countNonZero(mask)
def cropSun(img):
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
class image:
    def getFilename(path):
        img = []
        for filename in os.listdir(path):
            img.append(os.path.join(path,filename))
        return img

    