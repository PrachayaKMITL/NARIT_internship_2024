from src.preprocessing import thresholding,preprocessData
from src.TotalCalculation import timeConvertion,SunPosition
from src.ClassPrediction import prediction
import shutil,os
import cv2

mode = str(input("Enter writing mode(day/night) : "))
output_directory = str(input("Enter output directory : "))
start_date = '2024-01-01'
end_date = os.path.basename(image_directory)
location = [18.849417,98.9538]
days = timeConvertion().time_duration(start_date,end_date,include_end_date=True).days

LSTM = SunPosition.LSTM(time_zone_offset=7)
EoT = SunPosition.calculate_EoT(day=days)
TC = SunPosition.TimeCorrectionFactor(Longitude=location[1],LSTM=LSTM,EoT=EoT)
dec = SunPosition.declination(day=days)
suntime = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)
sunrise1,sunset1 = SunPosition.DaytimeInfo(latitude=location[0],declination=dec,TC=TC)

#Execution
image_directory = image_directory
mask = cv2.imread(mask_directory,cv2.IMREAD_GRAYSCALE)
mask = preprocessData().crop_center(mask,crop_size=570)
images,name = preprocessData().load_images_and_preprocess(path=image_directory,mask=mask,apply_crop_sun=True)
final,_,_ = thresholding().RBratio(input=images,filename=name,Time_zone=7,sunrise=sunrise1,sunset=sunset1)
filtering = lambda x : (x > sunrise1) & (x < sunset1)
decimal = [timeConvertion().datetime_to_decimal(time=timeConvertion().ticks_to_datetime(ticks=t,time_zone=7)) for t in name]
if mode == 'day':
    day_indices = [index for index, value in enumerate(decimal) if filtering(value)]
    final_day = final[0:len(day_indices)]
    name_day = [name[i] for i in day_indices]
    percentage = [prediction().CloudRatio(i,mask=mask) for i in final_day]
    classifier = [prediction().classify_sky(i,r) for i,r in percentage]
    for i in range(len(classifier)):
        if classifier[i] == classes:
            original = os.path.join(image_directory,f"{name_day[i]}.png")
            shutil.copy2(original,output_directory)
if mode == 'night':
    day_indices = [index for index, value in enumerate(decimal) if not filtering(value)]
    final_day = final[0:len(day_indices)]
    name_day = [name[i] for i in day_indices]
    percentage = [prediction().CloudRatio(i,mask=mask) for i in final_day]
    classifier = [prediction().classify_sky(i,r) for i,r in percentage]
    for i in range(len(classifier)):
        if classifier[i] == classes:
            original = os.path.join(image_directory,f"{name_day[i]}.png")
            shutil.copy2(original,output_directory)