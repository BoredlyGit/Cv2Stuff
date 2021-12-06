import cv2
import numpy
import numpy as np

all_contours = False
use_hsv = True

img = cv2.imread("images/test_1.jpg")
# img = cv2.GaussianBlur(og_img, (3, 3), 0)
# No grayscale cause using color masks

if use_hsv:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, (20, 100, 20), (40, 255, 255))
else:
    # TODO: improve ranges to accommodate tests 3 & 4 (seems like upper limit needs increasing) (maybe hsv)
    lit_ball_mask = cv2.inRange(img, (1, 110, 110), (70, 220, 240))
    shadowed_ball_mask = cv2.inRange(img, (0, 50, 50), (25, 130, 130))
    mask = lit_ball_mask + shadowed_ball_mask


# try to detect edges of overlapping balls
ball_edges = cv2.GaussianBlur(cv2.Canny(img, 0, 250), (3, 3), 0)
ball_edges = numpy.logical_and(mask, ball_edges)
ball_edges.dtype = np.dtype("uint8")
ball_edges = ball_edges * 255
mask = mask - ball_edges

cv2.imshow("mask", mask)
cv2.imshow("ball_edges", ball_edges)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

for contour in contours:
    print(3.14 * cv2.minEnclosingCircle(contour)[1] ** 2, cv2.contourArea(contour), ((3.14 * cv2.minEnclosingCircle(contour)[1] ** 2)/10), (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2 - cv2.contourArea(contour)))

if not all_contours:
    # check size and shape
    contours = [contour for contour in contours if cv2.contourArea(contour) > 500 and (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2 - cv2.contourArea(contour) < (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2)/4) ]

circles = [cv2.minEnclosingCircle(contour) for contour in contours]

cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
for circle in circles:
    cv2.circle(img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (255, 0, 0), 2)

cv2.imshow("processed", img)
cv2.waitKey(0)
