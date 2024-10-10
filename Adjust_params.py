import json
import cv2
import numpy as np
from src.preprocessing import preprocessData 

def RBratiosingle(factor, crop, crop_night):
    # Get the current positions of the trackbars
    factor = cv2.getTrackbarPos('Factor', 'Interactive Window')
    factor_night = cv2.getTrackbarPos('Factor_night', 'Interactive Window')

    # Split the image into its channels
    R, G, B = cv2.split(crop)
    R_n, G_n, B_n = cv2.split(crop_night)

    # Calculate the ratio R/B for day and night images
    ratio = np.log1p(R / (B + 1e-5)) * (factor / 100)
    ratio = cv2.convertScaleAbs(ratio)
    _, final_mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ratio_night = np.log1p(R_n / (B_n + 1e-5)) * (factor_night / 100)
    ratio_night = cv2.convertScaleAbs(ratio_night)
    _, final_mask_night = cv2.threshold(ratio_night, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _,mask_count = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    # Create masked images
    masked = cv2.bitwise_and(crop, crop, mask=final_mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    masked_night = cv2.bitwise_and(crop_night, crop_night, mask=final_mask_night)
    masked_night = cv2.cvtColor(masked_night, cv2.COLOR_BGR2RGB)

    # Display cloud percentage for day and night images
    cloud_percentage_day = cv2.countNonZero(final_mask) / (mask_count) * 100
    cloud_percentage_night = cv2.countNonZero(final_mask_night) / (mask_count) * 100

    cv2.putText(masked, f"Cloud Percentage: {cloud_percentage_day:.2f}%", (60, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(masked_night, f"Cloud Percentage: {cloud_percentage_night:.2f}%", (60, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stack and display both images
    stack = np.hstack((masked, masked_night))
    cv2.imshow('Interactive Window', stack)

    return [factor / 100, factor_night / 100]

# Get the image path from the user
img_path = input("Enter image path for thresholding (day): ")
img_path_night = input("Enter image path for thresholding (night): ")

mask_path = r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\masks\Domestic observatories\mask_Astropark.png'
input_img = cv2.imread(img_path)
input_img_night = cv2.imread(img_path_night)

# Convert the images from BGR to RGB and apply the mask
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img_night = cv2.cvtColor(input_img_night, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
crop = cv2.bitwise_and(input_img, input_img, mask=mask)
crop_night = cv2.bitwise_and(input_img_night, input_img_night, mask=mask)

# Crop the images using preprocessData method
crop = preprocessData().crop_center(crop, crop_size=570)
crop_night = preprocessData().crop_center(crop_night, crop_size=570)

# Check if the images were loaded successfully
if crop is None or crop_night is None:
    print("Error: Could not load images. Please check the paths.")
    exit()

# Create a window and trackbars
cv2.namedWindow('Interactive Window')
cv2.createTrackbar('Factor', 'Interactive Window', 1, 300, lambda x: None)
cv2.createTrackbar('Factor_night', 'Interactive Window', 1, 300, lambda x: None)

# Main loop to display and update the images
while True:
    factor_values = RBratiosingle(1, crop, crop_night)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key press
        factor_values = RBratiosingle(1, crop, crop_night)
        factor_file = {
            "Factor": factor_values[0],
            "Factor_night": factor_values[1]
        }
        with open('RBratio.json', 'w') as f:
            json.dump(factor_file, f)
        break

# Clean up and destroy all windows
cv2.destroyAllWindows()
