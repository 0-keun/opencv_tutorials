import cv2

image = cv2.imread('./image/JackDaniels.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("BGR_image:",image[0][0])
print("RGB_image:",rgb_image[0][0])
print("Gray_image:",gray_image[0][0])

cv2.imshow("Image", image)
cv2.imshow("Gray_image", gray_image)
cv2.imshow("RGB_Image",rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()