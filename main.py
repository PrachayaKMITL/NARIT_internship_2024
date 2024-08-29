import json
import pandas as pd
import pickle
import cv2
import warnings
import os
from ClassPrediction import prediction, visualizer
from TotalCalculation import timeConvertion, SunPosition
from preprocessing import image

config_path = str(input("Enter configuration file path : "))
with open(config_path, 'r') as config_file:
    config = json.load(config_file)
print(f"Current selection {config_path} selected")
# Extract parameters from the config file
start_date = config['parameters']['start_date']
image_size = config['parameters']['size']
mask_path = config['parameters']['mask']
timezone = config['parameters']['timezone']

# Extract paths from the config file
output_dir = config['paths']['output_directory']
kmean_model_path = config['paths']['kmean_model']
minik_model_path = config['paths']['minik_model']
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
print(f"Current configuration",config)
for i in image_list:
    m += 1
    time = int(os.path.splitext(os.path.basename(i))[0])
    warnings.filterwarnings("ignore")
    sunrise, sunset = SunPosition().SunriseSunset(filename=time, start_date=start_date, include_end_date=True)
    output = pred.total_prediction(image_path=i, mask_path=mask_path, sunrise=sunrise, sunset=sunset, kmeans=kmean, miniBatchesKmeans=minik)
    time = int(os.path.splitext(os.path.basename(i))[0])
    time = tim.ticks_to_datetime(time, time_zone=timezone)
    time = time.strftime('%Y-%m-%d %H:%M')
    pred_t = [output[0], output[1]]
    clarity = 100 - pred.weighted_prediction(weight=None, predicted_result=pred_t, cloud_percent=output[2], sky_status=output[3])[0]
    img = cv2.imread(i)
    raw = viz.image_to_base64(img)
    raw_final = viz.image_html(raw, size=image_size)
    image_base64 = viz.image_to_base64(output[4][0])
    final_image_html = viz.image_html(image_base64, size=[200,200])
    result.append([time, output[0][0], output[1][0], output[2], output[3], clarity, raw_final, final_image_html])
    d_out = pd.DataFrame({
        'Pred_1': [output[0]],
        'Pred_2': [output[1]],
        'Cloud_coverage': [output[2]],
        'Sky_status': [output[3]]
    })  
    viz.progress_bar(m, leng, 100)

print("\n---------Prediction complete---------")

df_out = pd.DataFrame(data=result, columns=['Time', 'Kmean_clustering', 'GMM_clustering',
                                            'Cloud_coverage %', 'Sky_status',
                                            'Sky clarity (%)', 'Raw image', 'Final image'])

df_out.to_html(os.path.join(output_dir, "Output.html"), index=False, escape=False, justify='center')
df_out.to_csv(os.path.join(output_dir, "output.csv"))

print("-----------Writing complete----------")
