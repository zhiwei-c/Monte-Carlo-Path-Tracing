import cv2
img = cv2.imread("leaf.png", -1)
apha_channel = img[:, :, 3]
_ = 1
