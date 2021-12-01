import cv2
import numpy
import sys

img = cv2.imread("images/ball.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


cv2.imshow("test", img)
cv2.waitKey(0)
