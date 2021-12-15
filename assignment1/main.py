import cv2
import numpy as np
import os


# for img_name in os.listdir("images"):
# print(img_name)

og_img = cv2.imread(f"images/bike-sgn.jpg")
# print(img)

og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2HSV)
img = cv2.GaussianBlur(og_img, (3, 3), 0)

# cv2.imshow("e", img)
img = cv2.inRange(img, (23, 104, 115), (27, 209, 253))
# img = np.bitwise_not(img)
# img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
print(img)
img = cv2.dilate(img, np.ones((3, 3), np.uint8))

# corners = cv2.cornerHarris(img, 20, 3, 0.04)
# print(corners.shape, img.shape)
# corners = cv2.threshold(corners, max(corners.flatten())*0.2, 255, cv2.THRESH_BINARY)[1]
# print(corners, corners.shape, img.shape)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
print(contours)

cv2.drawContours(og_img, contours, -1, (255, 255, 255), 1)
cv2.imshow("e", img)
cv2.imshow("ee", og_img)
cv2.waitKey(0)
