import cv2
import json
import shutil 
from src.preprocessing import thresholding, preprocessData
from src.TotalCalculation import timeConvertion, SunPosition
from src.ClassPrediction import prediction
import os

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-06-30'
location = [18.849417, 98.9538]
json_file_path = r"C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\RBratio.json"
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
factor = [data['Factor'],data['Factor_night']]
days = timeConvertion().time_duration(start_date, end_date, include_end_date=True).days

# Create SunPosition instance
LSTM = SunPosition.LSTM(time_zone_offset=7)
EoT = SunPosition.calculate_EoT(day=days)
TC = SunPosition.TimeCorrectionFactor(Longitude=location[1], LSTM=LSTM, EoT=EoT)
dec = SunPosition.declination(day=days)
suntime = SunPosition.DaytimeInfo(latitude=location[0], declination=dec, TC=TC)
sunrise1, sunset1 = SunPosition.DaytimeInfo(latitude=location[0], declination=dec, TC=TC)

def copy_categorical_day(image_directory, output_directory, mask_directory, classes, mode):
    mask = cv2.imread(mask_directory, cv2.IMREAD_GRAYSCALE)
    mask = preprocessData().crop_center(mask, crop_size=570)
    # Load and preprocess images
    images, name = preprocessData().load_images_and_preprocess(path=image_directory, mask=mask, apply_crop_sun=True)
    final, _, _ = thresholding().RBratio(input=images, factor=factor, filename=name, Time_zone=7, sunrise=sunrise1, sunset=sunset1)
    decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t, time_zone=7)) for t in name]
    filtering = lambda x: (x > sunrise1) & (x < sunset1)
    if mode == 'day':
        day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
    else: 
        day_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
    final_day = final[0:len(day_indices)]
    name_day = [name[i] for i in day_indices]
    percentage = [prediction().CloudRatio(i, mask=mask) for i in final_day]
    classifier = [prediction().classify_sky(i, r) for i, r in percentage]
    for i in range(len(classifier)):
        if classifier[i] in classes:
            original = os.path.join(image_directory, f"{name_day[i]}.png")
            shutil.copy2(original, output_directory)
def process_image_folders(main_directory):
    classes_map = {
        'Clear sky (0 oktas)': 'Clear',
        'Fewer clouds (1 okta)': 'Clear',
        'Few clouds (2 oktas)': 'Partly Cloudy',
        'Scatter (3 oktas)': 'Partly Cloudy',
        'Mostly Scatter (4 oktas)': 'Mostly Cloudy',
        'Partly Broken (5 oktas)': 'Mostly Cloudy',
        'Mostly Broken (6 oktas)': 'Cloudy',
        'Broken (7 oktas)': 'Cloudy',
        'Overcast (8 oktas)': 'Overcast',
    }

    for subdir, _, files in os.walk(main_directory):
        # Only process if there are images in the subdirectory
        if files:
            # Take the first image in the subdirectory to calculate sunrise and sunset
            first_image = files[0]
            image_path = os.path.join(subdir, first_image)
            
            # Calculate sunrise and sunset values for this specific folder
            # Use the previous calculations based on your logic
            # Process day mode (you can adjust the output directory based on the class)
            for cloud_class, output_suffix in classes_map.items():
                output_directory = os.path.join(r'C:\Users\ASUS\Documents\NARIT_internship_data\Dataset\image_data_day', output_suffix)
                os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

                copy_categorical_day(subdir, output_directory, mask_directory=r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\masks\Domestic observatories\mask_Astropark.png', 
                                     classes=cloud_class, mode='day')

                print(f"Processed {cloud_class} images from {subdir}")

# Define the main image directory
main_image_directory = [r'C:\Users\ASUS\Documents\NARIT_internship_data\All_sky_camera_Astropark_Chaingmai\2024-06',r'C:\Users\ASUS\Documents\NARIT_internship_data\All_sky_camera_Astropark_Chaingmai\2023-12']
for i in main_image_directory:
    process_image_folders(i)
