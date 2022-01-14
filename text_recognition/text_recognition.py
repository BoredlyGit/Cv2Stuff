import cv2
import pytesseract
import numpy as np
from PIL import Image


def connect_points(img, points):
    points = points.astype("uint64")
    prev_point = None
    first_point = None
    print(points)

    for point in points:
        if prev_point is None:
            prev_point = point
            first_point = point
            continue
        else:
            cv2.line(img, prev_point, point, (255, 0, 0))
            prev_point = point
            # cv2.line(img, point, first_point, (0, 255, 255))
    cv2.line(img, prev_point, first_point, (255, 0, 0))  # inplace

img = cv2.imread("images/capture2.PNG")
img = cv2.GaussianBlur(img, (5, 5), 10)
min_area = 500

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.inRange(img, np.array([99, 110, 71]), np.array([111, 255, 255]))

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda cnt: cv2.contourArea(cnt) >= min_area, contours))
blue_areas = [cv2.minAreaRect(cnt) for cnt in contours]


# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#
# print(blue_areas)
# for i, area in enumerate(blue_areas):
#     print(area)
#     angle = area[-1]
#
#
#     # TODO: This is cropping the image - make it not
#     image_center = tuple(np.array(img.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rotated = cv2.warpAffine(img, rot_mat, img.shape[:2], flags=cv2.INTER_LINEAR)

    # print("=" * 100)
    # print(pytesseract.image_to_boxes(Image.fromarray(rotated), config="--psm 7"))
    # print("=" * 100)
#     contours, hierarchy = cv2.findContours(rotated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # this will return multiple cntours???? take bounding box and crop main image first
#
#     # find child (internal) contours, draw a bounding box, and see if they are text, case tesseract can't do it on only the rotated image.
#     children = []
#     for relation in hierarchy:
#         if relation[2] != -1:  # has a child
#             hierarchy[relation[2]]  # the child
#         # x, y, w, h = cv2.boundingRect(contour)
#         # cropped = rotated[y:y+h, x:x+w]
#         # cv2.imshow("crop", cropped)
#         # print(pytesseract.image_to_string(rotated, config="--psm 6"))
#         # print("g")
#     cv2.imshow("rotat", rotated)
#
#     connect_points(img, cv2.boxPoints(area))
#     print("f")


print("=" * 100)
print(pytesseract.image_to_boxes(Image.fromarray(img), config="--psm 7"))
print("=" * 100)

cv2.imshow("original", img)
cv2.waitKey(0)
