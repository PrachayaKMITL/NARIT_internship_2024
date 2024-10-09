import json
import pickle
from src.preprocessing import preprocessData 
import cv2
import numpy as np

def RBratiosingle(factor):
    # Get the current positions of the trackbars
    factor = cv2.getTrackbarPos('Factor', 'Interactive Window')
    contrast = cv2.getTrackbarPos('Contrast','Interactive Window') / 100
    grid_size = cv2.getTrackbarPos('Grid Size', 'Interactive Window')
    # Split the image into its channels
    R, G, B = cv2.split(crop)
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(grid_size,grid_size))
    B = clahe.apply(B)

    # Calculate the ratio R/B
    ratio = np.log1p(R / (B + 1e-5)) * (factor/100)
    ratio = cv2.convertScaleAbs(ratio)

    # Apply thresholding to create a mask
    _, final_mask = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create a masked image
    masked = cv2.bitwise_and(crop, crop, mask=final_mask)
    masked = cv2.cvtColor(masked,cv2.COLOR_BGR2RGB)

    # Display the masked image
    cv2.putText(masked,
            f"cloud percentage : {cv2.countNonZero(final_mask)/cv2.countNonZero(mask)*100} %",
            org=(60, 40),  # Text start position
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Correct position for the font face
            fontScale=1,  # Font scale
            color=(255, 255, 255),  # White text
            thickness=2)  # Thickness of the text
    cv2.imshow('Final image', masked)
    cv2.imshow('Interactive Window', final_mask)
    return factor/100

# Get the image path from the user
img_path = input("Enter image path for thresholding: ")
mask_path = r'C:\Users\ASUS\Documents\NARIT_internship_2024\NARIT_internship_2024\masks\Domestic observatories\mask_Astropark.png'
input_img = cv2.imread(img_path)
# Convert the image from BGR to RGB
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
crop = cv2.bitwise_and(input_img,input_img,mask=mask)
crop = preprocessData().crop_center(crop,crop_size=570)
# Check if the image was loaded successfully
if input_img is None:
    print("Error: Could not load image. Please check the path.")
    exit()
# Create a window
cv2.namedWindow('Interactive Window')

# Create trackbars for factor and threshold
cv2.createTrackbar('Factor', 'Interactive Window', 1, 300, RBratiosingle)  # Default: 1, Max: 10
cv2.createTrackbar('Contrast', 'Interactive Window', 1, 200, RBratiosingle)
cv2.createTrackbar('Grid Size', 'Interactive Window', 1, 10, RBratiosingle)
# Call the function to initialize the display
value = RBratiosingle(1)
# Main loop to wait for user input
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key press
        value = cv2.getTrackbarPos('Factor','Interactive Window') / 100
        with open('RBratio.json','w') as f:
            json.dump({"Factor" : value},f)
        break
# Clean up and destroy all windows
cv2.destroyAllWindows()
