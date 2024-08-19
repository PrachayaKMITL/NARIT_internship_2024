from ClassPrediction import prediction
import pandas as pd
import pickle 
import cv2 
import numpy as np

with open('kmean_model_1.pkl', 'rb') as file:
    kmean = pickle.load(file)
with open('gmm_model_1.pkl', 'rb') as file:
    gmm = pickle.load(file)
path = input("Enter image path :")
data_path = 'GLCM_feature_ALL_sky_6June_dis3_ang45_test.csv'
df = pd.read_csv(data_path)
pred = prediction()
pred_1,pred_2,cloud_percentage,sky_status,final = pred.total_prediction(image_path=path,mask_path='mask_delete_5.png',kmeans=kmean,GMM=gmm,df=df)
image = cv2.imread(path)
pred_t = [pred_1,pred_2]
print(f"Rain risk : {pred.weighted_prediction(weight=None,predicted_result=pred_t,cloud_percent=cloud_percentage,sky_status=sky_status)}")
cv2.putText(image, f"Cloud Coverage: {cloud_percentage:.2f}%", 
            (10, 30),  
            cv2.FONT_HERSHEY_DUPLEX, 
            0.7,
            (255, 255, 255), 
            1)  
cv2.putText(image, f"Prediction_1: {pred_1}", 
            (10, 100),  
            cv2.FONT_HERSHEY_DUPLEX, 
            0.7,
            (255, 255, 255), 
            1)  

cv2.putText(image, f"Sky Status: {sky_status}", 
            (10, 70), 
            cv2.FONT_HERSHEY_DUPLEX,  
            0.7,  
            (255, 255, 255),  
            1)
cv2.putText(image, f"Prediction_2: {pred_2}", 
            (10, 130),  
            cv2.FONT_HERSHEY_DUPLEX, 
            0.7,
            (255, 255, 255), 
            1)
cv2.imshow('Prediction', image)
cv2.imshow('Final',final[0])
cv2.waitKey(0)
cv2.destroyAllWindows()