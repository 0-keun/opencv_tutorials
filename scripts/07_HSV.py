import cv2
import numpy as np

image_path = './image/road.jpg'
image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

yellow_lower = np.array([20,100,100])
yellow_upper = np.array([30,255,255])
yellow_mask  = cv2.inRange(hsv_image,yellow_lower,yellow_upper)

white_lower = np.array([0,0,200])
white_upper = np.array([180,25,255])
white_mask  = cv2.inRange(hsv_image,white_lower,white_upper)

combined_mask = cv2.bitwise_or(yellow_mask,white_mask)
combined_image = cv2.bitwise_and(image,image,mask=combined_mask)

cv2.imshow('Original_image',image)
cv2.imshow('Combined_image',combined_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()
