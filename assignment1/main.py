import random

import cv2
import numpy as np
import os

# Try to avoid color dependence.
# TODO: find a away to bridge gaps... Maybe HoughLineTransform
og_img = cv2.imread(f"images/stop-sign.jpg")

img = cv2.GaussianBlur(og_img, (3, 3), 0)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img.dtype = "int32"
img = cv2.Canny(img, 210, 255)

#
# img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((3, 3), "uint8"))
# line_detector = cv2.ximgproc.createFastLineDetector(30)
# img = line_detector.drawSegments(canny, line_detector.detect(img))
#

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda ctr: cv2.contourArea(ctr) > 500, contours))
print(contours)
# img.dtype = "uint8"
# contours *= 4

for i, contour in enumerate(contours):
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    print(cv2.contourArea(contour), color)
    cv2.drawContours(og_img, contours, i, color, thickness=cv2.FILLED)
cv2.drawContours(og_img, contours, -1, (0, 255, 0), 1)
cv2.imshow("E", og_img)
cv2.imshow("f", img)
cv2.waitKey(0)
