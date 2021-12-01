import cv2
import numpy
import sys

og_img = cv2.imread("images/test_1.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.GaussianBlur(og_img, (7, 7), 0)

img = cv2.inRange(img, (1, 110, 110), (70, 220, 240))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(contours)
# TODO: Filter contours by shape
cv2.drawContours(og_img, contours, -1, (0, 255, 0), 2)

cv2.imshow("processed", img)

cv2.imshow("og", og_img)
cv2.waitKey(0)
