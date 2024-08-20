import cv2
import numpy as np

image_path = './image/road.jpg'
original_image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)

yellow_lower = np.array([20,100,100])
yellow_upper = np.array([30,255,255])

white_lower = np.array([0,0,200])
white_upper = np.array([180,25,255])

def yellow_filter(image):
    yellow_mask  = cv2.inRange(hsv_image,yellow_lower,yellow_upper)
    yellow_image = cv2.bitwise_and(image,image,mask=yellow_mask)
    cv2.imshow('yellow_image',yellow_image)
    return yellow_image

def white_filter(image):
    white_mask  = cv2.inRange(hsv_image,white_lower,white_upper)
    white_image = cv2.bitwise_and(image,image,mask=white_mask)
    cv2.imshow('white_image',white_image)
    return white_image

def combine_filter(image):
    yellow_mask = cv2.inRange(hsv_image,yellow_lower,yellow_upper)
    white_mask = cv2.inRange(hsv_image,white_lower,white_upper)
    combined_mask = cv2.bitwise_or(yellow_mask,white_mask)
    combined_image = cv2.bitwise_and(image,image,mask=combined_mask)
    cv2.imshow('combined_image',combined_image)
    return combined_image

yellow_image = yellow_filter(original_image)
white_image = white_filter(original_image)
combined_image = combine_filter(original_image)

cv2.imshow('original_image',original_image)
cv2.waitKey(0) 
cv2.destroyAllWindows()