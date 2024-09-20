import json
import pandas as pd
import pickle
import cv2
import warnings
import os
import time as timer 
from ClassPrediction import prediction, visualizer
import ClassPrediction
from TotalCalculation import timeConvertion, SunPosition
from preprocessing import image

Mode = str(input("Enter daytime mode (Day/Night) : "))
sky_cam = str(input("Enter configuration selection(Chile,Astropark,Astropark_december,China) : "))
with open("Configuration//configuration.json", 'r') as config_file:
    config = json.load(config_file) 
print(f"Location : '{sky_cam}'")
# Extract parameters from the config file
start_date = config[Mode][sky_cam]['parameters']['start_date']
image_size = config[Mode][sky_cam]['parameters']['size']
mask_path = config[Mode][sky_cam]['parameters']['mask']
timezone = config[Mode][sky_cam]['parameters']['timezone']
location = config[Mode][sky_cam]['parameters']['location']
print("Start date : ", start_date)
print("Time zone : ", timezone)
print("Mask path : ", mask_path)
print(f"Coordinate : {location[0]}°N {location[1]}°E")
print("#################################################")
# Extract paths from the config file
output_dir = config[Mode][sky_cam]['paths']['output_directory']
kmean_model_path = config[Mode][sky_cam]['paths']['kmean_model']
minik_model_path = config[Mode][sky_cam]['paths']['minik_model']
m = 0
with open(kmean_model_path, 'rb') as file:
    kmean = pickle.load(file)
with open(minik_model_path, 'rb') as file:
    minik = pickle.load(file)
result = []
pred = prediction()
viz = visualizer()
tim = timeConvertion()
image_dir = str(input("Enter image directory : "))
image_list = image.getFilename(image_dir)
leng = len(image_list)
start = timer.time()
for i in image_list:
    m += 1
    time = int(os.path.splitext(os.path.basename(i))[0])
    warnings.filterwarnings("ignore")
    sunrise, sunset = SunPosition().SunriseSunset(location=location,filename=time, start_date=start_date,Time_zone=timezone, include_end_date=True)
    output = pred.total_prediction(image_path=i, mask_path=mask_path, sunrise=sunrise-0.067, sunset=sunset+0.067, kmeans=kmean, miniBatchesKmeans=minik)
    time = int(os.path.splitext(os.path.basename(i))[0])
    time = tim.ticks_to_datetime(time, time_zone=timezone)
    time = time.strftime('%Y-%m-%d %H:%M')
    pred_t = [output[0], output[1]]
    clarity = 100 - pred.weighted_prediction(weight=None, cloud_percent=output[2], red_channel=output[5][3], Blue_channel=output[5][4])
    img = cv2.imread(i)
    sky_status = prediction().sky_status(output[2])
    cv2.putText(output[4],f"Cloud percentage : {output[2]}",(20,40),cv2.FONT_HERSHEY_COMPLEX,
                0.5,(255,255,255),1)
    cv2.putText(output[4],f"Sky status : {sky_status}",(20,70),cv2.FONT_HERSHEY_COMPLEX,
                0.5,(255,255,255),1)
    raw = viz.image_to_base64(img)
    raw_final = viz.image_html(raw, size=image_size)
    image_base64 = viz.image_to_base64(output[4])
    final_image_html = viz.image_html(image_base64, size=[200,200])
    result.append([time, int(output[0][0]), int(output[1][0]),output[2], output[3], clarity, sky_status, raw_final, final_image_html]) 
    viz.progress_bar(m, leng, 100)

print("\n---------Prediction complete---------")

df_out = pd.DataFrame(data=result, columns=['Time', 'Kmean_clustering', 'GMM_model',
                                            'Cloud_coverage %', 'Cloud_status',
                                            'Sky clarity (%)','sky_status','Raw image', 'Final image'])

df_out.to_html(os.path.join(output_dir, f"{sky_cam}_Output.html"), index=False, escape=False, justify='center')
df_out = df_out.drop(columns=['Final image','Raw image'])
df_out.to_csv(os.path.join(output_dir, f"{sky_cam}_Output.csv"))

print("-----------Writing complete----------\n")
print(f"Runtime : {timer.time() - start} Seconds")
if str(input("Clear console? (yes/no) : ")) == 'yes':
    os.system('cls')
#Estimate Big O
#O(n) = 0.088*n (seconds)