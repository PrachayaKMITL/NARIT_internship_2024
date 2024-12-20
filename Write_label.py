import cv2
import json
import shutil
import sys, os
from src.preprocessing import thresholding, preprocessData
from src.TotalCalculation import timeConvertion, SunPosition
from src.ClassPrediction import prediction

# Define the start and end dates
start_date = '2024-01-01'
end_date = '2024-09-10'
location = [18.849417,98.9538]
#json_file_path = r"C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\RBratio.json"
#with open(json_file_path, 'r') as json_file:
#    data = json.load(json_file)
#factor = [data['Factor'], data['Factor_night']]
days = timeConvertion().time_duration(start_date, end_date, include_end_date=True).days

# Create SunPosition instance
LSTM = SunPosition.LSTM(time_zone_offset=7)
EoT = SunPosition.calculate_EoT(day=days)
TC = SunPosition.TimeCorrectionFactor(Longitude=location[1], LSTM=LSTM, EoT=EoT)
dec = SunPosition.declination(day=days)
sunrise1, sunset1 = SunPosition.DaytimeInfo(latitude=location[0], declination=dec, TC=TC)
print(f"sunrise : {sunrise1}")
print(f"sunset : {sunset1}")
def copy_categorical_day(image_directory, output_directory, mask_directory, classes:dict, mode):
    mask = cv2.imread(mask_directory, cv2.IMREAD_GRAYSCALE)
    # Load and preprocess images
    images, name = preprocessData().load_images_and_preprocess(path=image_directory, mask=mask, apply_crop_sun=False)
    name = [int(i) for i in name]
    final = []
    for i in images:
        R,G,B = cv2.split(i)
        B = B + 1e-3
        RB = (R/B) * 1
        RB = cv2.convertScaleAbs(RB)
        i = cv2.bitwise_and(i,i,mask=RB)
        i = cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)
        final.append(i)

    decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t, time_zone=7)) for t in name]
    filtering = lambda x: (x > sunrise1) & (x < sunset1)
    if mode == 'day':
        day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
        
    if mode == 'night': 
        day_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
    final_day = [final[i] for i in day_indices]
    name_day = [name[i] for i in day_indices]
    
    # Calculate cloud coverage and classify the images
    percentage = [prediction().CloudRatio(i, mask=mask)[0] for i in final_day]
    classifier = [prediction().classify_sky(i) for i in percentage]

    # For each classified image, move it to the appropriate class folder
    for idx, cloud_class in enumerate(classifier):
        if cloud_class in classes:
            # Define the target folder based on the classification
            target_folder = os.path.join(output_directory, classes[cloud_class])
            os.makedirs(target_folder, exist_ok=True)  # Ensure the folder exists
            
            # Move the file to the target folder
            image_filename = name_day[idx]
            source_path = os.path.join(image_directory, str(image_filename) + '.png')
            write_image = cv2.imread(source_path)
            if write_image.shape[0] != 705 or write_image.shape[1] != 1036:
                write_image = cv2.resize(write_image, (1036, 705))
            write_path = os.path.join(target_folder,str(image_filename)+'.png')
            cv2.imwrite(write_path,write_image)
def count_files_in_folder(directory):
    return sum(os.path.isfile(os.path.join(directory, f)) for f in os.listdir(directory))
def process_image_folders(main_directory):
    # Define class labels to folder names
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
    
    # Iterate through subdirectories and process images
    for subdir, _, _ in os.walk(main_directory):
        copy_categorical_day(subdir, output_directory, 
                             mask_directory=r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\masks\Domestic observatories\Mask_TNO.png', 
                             classes=classes_map, mode='night')
        print(f"Processed images from {subdir}")

# Define the main image directory
main_image_directory = r'C:\Users\ASUS\Documents\NARIT_internship_data\All_sky_camera_TNO\2024-09'
output_directory = r'C:\Users\ASUS\Documents\NARIT_internship_data\Dataset\Image_data_TNO\Image_data_Night'
for i in os.listdir(main_image_directory):
    image_data_path = os.path.join(main_image_directory, i)
    process_image_folders(image_data_path)
    
for foldername  in os.listdir(output_directory):
    files = count_files_in_folder(os.path.join(output_directory,foldername))

