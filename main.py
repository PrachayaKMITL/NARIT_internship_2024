from ClassPrediction import prediction,visualizer
from preprocessing import image
from IPython.display import HTML, display
import pandas as pd
import pickle 
import cv2 
import numpy as np
import warnings
m = 0
with open(r'models\kmean_model_2.pkl', 'rb') as file:
    kmean = pickle.load(file)
with open(r'models\gmm_model_2.pkl', 'rb') as file:
    gmm = pickle.load(file)
result = []
pred = prediction()
viz = visualizer()
image_list = image.getFilename(r'C:\Users\ASUS\Documents\NARIT_internship_data\All_sky_camera_Astropark_Chaingmai\2024-06\2024-06-16')
leng = len(image_list)
for i in image_list:
    m += 1
    warnings.filterwarnings("ignore")
    output = pred.total_prediction(image_path=i,mask_path='mask_delete_5.png',kmeans=kmean,GMM=gmm)
    pred_t = [output[0],output[1]]
    clarity = 100-pred.weighted_prediction(weight=None,predicted_result=pred_t,cloud_percent=output[2],sky_status=output[3])[0]
    img = cv2.imread(i)
    raw= viz.image_to_base64(img)
    raw_final = viz.image_html(raw,size=[285,200])

    image_base64 = viz.image_to_base64(output[4][0])
    final_image_html = viz.image_html(image_base64,size=[200,200])
    result.append([output[0][0], output[1][0], output[2], output[3],clarity,raw_final,final_image_html])
    d_out = pd.DataFrame({
        'Pred_1': [output[0]],
        'Pred_2': [output[1]],
        'Cloud_coverage': [output[2]],
        'Sky_status': [output[3]]
    })
    viz.progress_bar(m, leng, 100)
print("\n---------Prediction complete---------")
df_out = pd.DataFrame(data=result,columns=['Kmean_clustering','GMM_clustering',
                                           'Cloud_coverage %','Sky_status',
                                           'Sky clarity','Raw image','Final image'])
df_out.to_html(r"C:\Users\ASUS\Documents\NARIT_internship_data\Output_HTML\Output.html",index=False,escape=False,justify='center')
