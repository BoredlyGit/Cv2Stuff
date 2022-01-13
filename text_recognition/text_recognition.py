import cv2
import pytesseract
import numpy as np

img = cv2.imread("images/capture2.PNG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.inRange(img, np.array([99, 110, 71]), np.array([111, 255, 255]))

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow("", img)


print(pytesseract.image_to_string(img))
cv2.waitKey(0)
