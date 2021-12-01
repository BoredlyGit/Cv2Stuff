import cv2
import numpy
import sys
from pprint import pprint

img = cv2.imread("images/ball.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

r, g, b = cv2.split(img)
# print(r, type(r))
f = [r, g, b]

img = cv2.merge(f)
print(img, type(img), img[1299].shape)

cv2.imshow("test", img)
cv2.waitKey(0)
