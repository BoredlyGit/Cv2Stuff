import random
import cv2
import numpy as np
import os

# Try to avoid color dependence.
# TODO: find a way to bridge gaps... Maybe HoughLineTransform

for img_path in os.listdir("images/"):
    og_img = cv2.imread(f"images/{img_path}")
    # og_img = cv2.imread(f"images/road-closed-sign.jpg")

    img = cv2.GaussianBlur(og_img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, 187, 255)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((1, 1)))
    cv2.imshow("canny", img)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min size and attempt to filter out leaves & large noise with len() (more vertices -> more points)
    contours = list(filter(lambda ctr: cv2.contourArea(ctr) > 700 and len(ctr) < 850, contours))
    # print([len(contour) for contour in contours])
    for i, contour in enumerate(contours):
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        approx = cv2.approxPolyDP(contour, 30, True)  # 30 is literally a random number i have no idea why it works
        cv2.drawContours(og_img, contours, i, (0, 255, 0), 2)
        if len(approx) > 2:
            print(approx, len(approx))
            moment = cv2.moments(approx)
            cv2.drawContours(og_img, [approx], 0, color, thickness=cv2.FILLED)
            cv2.putText(og_img, f"{len(approx)}-sided shape", (int(moment["m10"]/moment["m00"]), int(moment["m01"]/moment["m00"])), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255))

    cv2.imshow("E", og_img)
    cv2.waitKey(0)
