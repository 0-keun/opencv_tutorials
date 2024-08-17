import cv2

image = cv2.imread('./image/JackDaniels.jpg')

print(type(image))
print(image)

cv2.imshow("Bottle Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()