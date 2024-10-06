import cv2
import numpy as np
"""
This code is for mask creation. These are steps of using this code
    1. Select image that has the best view of sky (Considering the most contrast between sky and obstacle. Sky must be bright while obstacle area are mostly dark)
    2. Paste image path into terminal without "".
    3. Run the code by press ENTER on the keyboard. Interactable windows will appears.
    4. Adjust value as needed, then press 's' to save
    5. Image will be saved in the root folder.
------ Usage of the mask--------
    Masked is use for improve quality of the algorithm that calculate values. Since some all sky image have some glare from the ground in the night.
    It might affect the quality of the model if any disturbance exist in the image. User may not need mask if user think that mask is unnescessary.
    but result may vary.
"""
processed_image = None #define parameters processed_image 

def update(val):
    global processed_image # Declare image as global parameter
    # Get current positions of the trackbars
    thresh_val = cv2.getTrackbarPos('Threshold', 'Interactive Window')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Interactive Window')
    iteration = cv2.getTrackbarPos('Iteration', 'Interactive Window')
    kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1

    # Apply binary threshold
    _, thresh = cv2.threshold(clahe_image, thresh_val, 255, cv2.THRESH_BINARY)

    # Define a kernel with the selected size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological operations
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(opening, kernel, iterations=iteration)
    erosion = cv2.erode(dilated, kernel, iterations=iteration+2)

    processed_image = erosion
    
    masked_image = cv2.bitwise_and(img,img,mask=erosion)
    downstack = np.hstack((dilated, erosion))
    cv2.imshow('Interactive Window', downstack)
    cv2.imshow('Masked image', masked_image)
#load image by images name
path = input("Enter image path: ")
img = cv2.imread(path)
if img is None:
    print("Error: Unable to read image.")
    exit(0)

B, G, R = cv2.split(img)
clahe = cv2.createCLAHE(clipLimit=0.001, tileGridSize=(7,7))
clahe_image = clahe.apply(B)

# Create a window
cv2.namedWindow('Interactive Window')

# Create trackbars for threshold and kernel size
cv2.createTrackbar('Threshold', 'Interactive Window', 55, 255, update)  # Default: 55, Max: 255
cv2.createTrackbar('Kernel Size', 'Interactive Window', 1, 10, update)  # Default: 1, Max: 10
cv2.createTrackbar('Iteration', 'Interactive Window', 1, 10, update)  # Default: 1, Max: 10

# Initial update to display the image
update(0)

# Wait for 'q' to quit
while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.waitKey(1) & 0xFF == ord('s') and processed_image is not None:
        cv2.imwrite('Mask.png',processed_image)
        print("image saved!")

# Destroy windows
cv2.destroyAllWindows()
