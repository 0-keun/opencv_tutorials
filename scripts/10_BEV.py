import cv2
import numpy as np
import math

PI = 3.1415926

frameWidth = 640
frameHeight = 720

def update_perspective(val):

    alpha = (cv2.getTrackbarPos("Alpha", "Result") - 90) * PI / 180
    beta = (cv2.getTrackbarPos("Beta", "Result") - 90) * PI / 180
    gamma = (cv2.getTrackbarPos("Gamma", "Result") - 90) * PI / 180
    focalLength = cv2.getTrackbarPos("f", "Result")
    dist = cv2.getTrackbarPos("Distance", "Result")

    image_size = (frameWidth, frameHeight)
    w, h = image_size

    A1 = np.array([[1, 0, -w / 2],
                [0, 1, -h / 2],
                [0, 0, 0],
                [0, 0, 1]], dtype=np.float32)

    RX = np.array([[1, 0, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha), 0],
                [0, math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                [0, 1, 0, 0],
                [math.sin(beta), 0, math.cos(beta), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                [math.sin(gamma), math.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    R = np.dot(np.dot(RX, RY), RZ)

    T = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dist],
                [0, 0, 0, 1]], dtype=np.float32)

    K = np.array([[focalLength, 0, w / 2, 0],
                [0, focalLength, h / 2, 0],
                [0, 0, 1, 0]], dtype=np.float32)

    transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)

    destination = cv2.warpPerspective(source, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    cv2.imshow("Result", destination)

def constant_perspective(val):
    alpha = (val[0]- 90) * PI / 180
    beta = (val[1]- 90) * PI / 180
    gamma = (val[2]- 90) * PI / 180
    focalLength = val[3]
    dist = val[4]

    image_size = (frameWidth, frameHeight)
    w, h = image_size

    A1 = np.array([[1, 0, -w / 2],
                [0, 1, -h / 2],
                [0, 0, 0],
                [0, 0, 1]], dtype=np.float32)

    RX = np.array([[1, 0, 0, 0],
                [0, math.cos(alpha), -math.sin(alpha), 0],
                [0, math.sin(alpha), math.cos(alpha), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                [0, 1, 0, 0],
                [math.sin(beta), 0, math.cos(beta), 0],
                [0, 0, 0, 1]], dtype=np.float32)

    RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                [math.sin(gamma), math.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    R = np.dot(np.dot(RX, RY), RZ)

    T = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dist],
                [0, 0, 0, 1]], dtype=np.float32)

    K = np.array([[focalLength, 0, w / 2, 0],
                [0, focalLength, h / 2, 0],
                [0, 0, 1, 0]], dtype=np.float32)

    transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)

    destination = cv2.warpPerspective(source, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    cv2.imshow("Constant_Result", destination)



source = cv2.imread('./image/road_camera.jpg')  # Replace with your image file path

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.namedWindow("Constant_Result", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Alpha", "Result", 90, 180, update_perspective)
cv2.createTrackbar("Beta", "Result", 90, 180, update_perspective)
cv2.createTrackbar("Gamma", "Result", 90, 180, update_perspective)
cv2.createTrackbar("f", "Result", 500, 2000, update_perspective)
cv2.createTrackbar("Distance", "Result", 500, 2000, update_perspective)


# get_parameters
update_perspective(0)

# # get_BEV_image
# constant_perspective([10,90,90,500,280])

cv2.waitKey(0)
cv2.destroyAllWindows()