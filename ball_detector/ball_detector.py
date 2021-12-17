import cv2
import numpy
import numpy as np
import json
import pprint

# TODO: record and return data
# TODO: try to detect blobs of balls somehow?
# Suggestion: use ellipses instead and check ratio?

img_path = "images/test_1.jpg"
img_path = img_path.lower()
all_contours = False
use_hsv = True
use_video = True


with open("profiles.json") as img_profiles:
    img_profiles = json.load(img_profiles)
img_profile = img_profiles["default"]
img_profile.update(img_profiles.get(img_path, {}))
print(f"Using profile: {img_profile['name']}\n{pprint.pformat(img_profile)}")


# img = cv2.imread(img_path)
if use_video:
    cap = cv2.VideoCapture(0)

while True:
    if use_video:
        _, og_img = cap.read()
        img = og_img.copy()

        h, w = img.shape[:2]
        center = (h//2, w//2)
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 180, 1.0), (w, h))
        og_img = cv2.warpAffine(og_img, cv2.getRotationMatrix2D(center, 180, 1.0), (h, w))
    else:
        og_img = cv2.imread(img_path)

    if img_profile["img_resize"]:
        img = cv2.resize(img, img_profile["img_resize"])
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # No grayscale cause using color masks

    if use_hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # doesnt accept tuples for some reason.
        mask = cv2.inRange(img, np.array(img_profile["ball_hsv_min"]), np.array(img_profile["ball_hsv_max"]))
    else:
        lit_ball_mask = cv2.inRange(img, (1, 110, 110), (70, 220, 240))
        shadowed_ball_mask = cv2.inRange(img, (0, 50, 50), (25, 130, 130))
        mask = lit_ball_mask + shadowed_ball_mask


    # try to detect edges of overlapping balls
    ball_edges = cv2.GaussianBlur(cv2.Canny(img, 0, 250), (3, 3), 0)
    ball_edges = numpy.logical_and(mask, ball_edges)
    ball_edges.dtype = np.dtype("uint8")
    ball_edges = ball_edges * 255
    mask = mask - ball_edges

    # cv2.imshow("mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not all_contours:
        # check size and shape, (see variables above)
        contours = [contour for contour in contours
                    if cv2.contourArea(contour) > img_profile["ball_min_area"]
                    and (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2 - cv2.contourArea(contour)
                         < (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2) * (1 - img_profile["ball_min_coverage"]))]

    circles = [cv2.minEnclosingCircle(contour) for contour in contours]

    cv2.drawContours(og_img, contours, -1, (255, 255, 255), 2)
    for circle in circles:
        cv2.circle(og_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (255, 0, 0), 2)

    cv2.imshow("", og_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
