import cv2
import numpy as np

path = input("Enter image path : ")
#path = r"C:\Users\ASUS\Documents\NARIT_internship_data\Test_folder\638635393872115938.png"
img = cv2.imread(path)
B,G,R = cv2.split(img)
clahe = cv2.createCLAHE(clipLimit=0.001, tileGridSize=(7,7))
clahe_image = clahe.apply(B)

rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
_,thresh = cv2.threshold(clahe_image,55,255,cv2.THRESH_BINARY)
kernel = np.array([
    [1, -1, 1],
    [1, -1, 1],
    [1, -1, 1]
])
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
dilated = cv2.dilate(thresh, kernel, iterations=1)
erosion = cv2.erode(opening, kernel, iterations=2)

Masked_out = cv2.bitwise_and(img,img,mask=erosion)

upstack = np.hstack((thresh,opening))
downstack = np.hstack((dilated,erosion))
whole_stack = np.vstack((upstack,downstack))


cv2.imshow("Stacked image",erosion)
cv2.imshow("Masked image",Masked_out)
cv2.waitKey(0)
cv2.destroyAllWindows()