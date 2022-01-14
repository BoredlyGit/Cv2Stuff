import cv2
import numpy as np
from PIL import Image
import pytesseract

# def connect_points(img, points):
#     points = points.astype("uint64")
#     prev_point = None
#     first_point = None
#     print(points)
#
#     for point in points:
#         if prev_point is None:
#             prev_point = point
#             first_point = point
#             continue
#         else:
#             cv2.line(img, prev_point, point, (255, 0, 0))
#             prev_point = point
#             # cv2.line(img, point, first_point, (0, 255, 255))
#     cv2.line(img, prev_point, first_point, (255, 0, 0))  # inplace


def crop(img, pt1, pt2):
    return img[pt1[1]:pt2[1], pt1[0]:pt2[0]]


img = cv2.imread("images/capture2.PNG")
img = cv2.GaussianBlur(img, (15, 15), 10)
min_area = 500

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.inRange(img, np.array([99, 110, 71]), np.array([111, 255, 255]))

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda cnt: cv2.contourArea(cnt) >= min_area, contours))
blue_areas = [{"rotated": cv2.minAreaRect(cnt), "bounding": cv2.boundingRect(cnt)} for cnt in contours]  # rotated rectangles
# blue_areas = [cv2.boundingRect(cnt) for cnt in contours]

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#
print(blue_areas)
for i, area in enumerate(blue_areas):
    print(area)
    x, y, w, h = area["bounding"]
    cropped = crop(img, (x, y), (x+w, y+h))
    print(cropped)
    cv2.imshow("cropped", cropped)
    # print("=" * 100)
    # print(pytesseract.image_to_boxes(Image.fromarray(cropped), config="--psm 7"))  # psm 7 = one-line text
    # print("=" * 100)

    angle = area["rotated"][-1]

    image_center = tuple([1000, 0])
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated = cv2.warpAffine(cropped, rot_mat, (cropped.shape[1], cropped.shape[0]), flags=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_x, final_y, final_w, final_h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    final = crop(rotated, (final_x, final_y), (final_x+final_w, final_y+final_h))

    boxes = map(lambda x: x.split(" "), pytesseract.image_to_boxes(Image.fromarray(final), config="--psm 7").split("\n")[:-1])
    print(tuple(boxes))
    final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    for char, x, y, w, h, _ in boxes:
        cv2.rectangle(final, (x, y), (x+w, y-h), (0, 255, 0))

    cv2.imshow("final", final)

# print("=" * 100)
# print(pytesseract.image_to_boxes(Image.fromarray(img), config="--psm 7"))  # psm 7 = one-line text
# print("=" * 100)

cv2.imshow("original", img)
cv2.waitKey(0)
