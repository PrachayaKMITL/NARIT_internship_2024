import numpy as np
import cv2
import os
from IPython.display import Image
from matplotlib import pyplot as plt
import shutil

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
            



    