import cv2
import numpy
import sys

og_img = cv2.imread("images/test_1.jpg")

img = cv2.GaussianBlur(og_img, (7, 7), 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) No grayscale cause using color masks

# SUGGESTION: take 2 ranges, 1 for the lit sections of the ball, and one for the shadowed section and combine?
lit_ball_mask = cv2.inRange(img, (1, 110, 110), (70, 220, 240))
shadowed_ball_mask = cv2.inRange(img, (0, 50, 50), (25, 130, 130))
mask = lit_ball_mask + shadowed_ball_mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
# TODO: Filter contours by shape
# TODO: Account for balls next to each other appearing as one contour
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)


cv2.imshow("processed", img)

# cv2.imshow("og", og_img)
cv2.waitKey(0)
