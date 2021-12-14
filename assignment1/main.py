import cv2
import numpy as np
import os

hsv_ranges = {
    "bike-sign.jpg": ((22, 101, 189), (51, 255, 255)),
    "pedestrian-sign.jpg": ((6, 69, 64), ())
}

for img_name in os.listdir("images"):
    print(img_name)
    img = cv2.imread(f"images/{img_name}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    # contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    #
    # cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
    cv2.imshow(img_name, img)
    cv2.waitKey(100000)
