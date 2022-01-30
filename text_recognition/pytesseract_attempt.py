import cv2
import numpy as np
import json
from PIL import Image
import pytesseract
import cProfile
import pstats
pytesseract.pytesseract.tesseract_cmd = "tesseract/tesseract"

with open("hsv_ranges.json", "r+") as hsv_ranges_json:
    hsv_ranges = json.load(hsv_ranges_json)


def crop(img, pt1, pt2):
    return img[pt1[1]:pt2[1], pt1[0]:pt2[0]]


def main():
    use_video = True
    img_path = "images/test_8.jpg"
    w, h = (360, 240)  # lower size if too slow

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        if use_video:
            _, img = cap.read()  # noqa
        else:
            img = cv2.imread(img_path)

        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        blue_mask, red_mask = np.zeros((h, w), dtype="uint8"), np.zeros((h, w), dtype="uint8")

        for hsv_range in hsv_ranges["blue_bumper"]:
            blue_mask = np.bitwise_or(blue_mask, cv2.inRange(img, np.array(hsv_range["min"]), np.array(hsv_range["max"])))

        for hsv_range in hsv_ranges["red_bumper"]:
            red_mask = np.bitwise_or(red_mask, cv2.inRange(img, np.array(hsv_range["min"]), np.array(hsv_range["max"])))

        combined_mask = np.bitwise_or(blue_mask, red_mask)
        cv2.imshow("red", red_mask)
        cv2.imshow("blue", blue_mask)
        cv2.imshow("mask", combined_mask)
        # maybe consider image_to_data? - could be useful but rn just add complexity for no real benefit
        # psm 11 = sparse text
        boxes = pytesseract.image_to_boxes(Image.fromarray(combined_mask), config="--psm 11", output_type=pytesseract.Output.DICT)

        if boxes:  # tesseract returns empty dict if no text
            for i in range(len(boxes["left"])):
                x1, y1, x2, y2 = boxes["left"][i], boxes["top"][i], boxes["right"][i], boxes["bottom"][i]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imshow("img", img)
        if use_video:
            if cv2.waitKey(1) in (27, 113):  # 27 = esc, 113 = q
                break
        else:
            cv2.waitKey(0)
            break


if __name__ == "__main__":
    cProfile.run('main()', filename="tesseract_profiler_out")
    stats = pstats.Stats("tesseract_profiler_out")
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


# min_area = 500
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # img = cv2.GaussianBlur(img, (5, 5), 0)
# img = cv2.inRange(img, np.array([99, 110, 71]), np.array([111, 255, 255]))
#
# contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = list(filter(lambda cnt: cv2.contourArea(cnt) >= min_area, contours))
# blue_areas = [{"rotated": cv2.minAreaRect(cnt), "bounding": cv2.boundingRect(cnt)} for cnt in contours]  # rotated rectangles
# # blue_areas = [cv2.boundingRect(cnt) for cnt in contours]
#
# # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
# #
# print(blue_areas)
# for i, area in enumerate(blue_areas):
#     print(area)
#     x, y, w, h = area["bounding"]
#     cropped = crop(img, (x, y), (x+w, y+h))
#     print(cropped)
#     cv2.imshow("cropped", cropped)
#     # print("=" * 100)
#     # print(pytesseract.image_to_boxes(Image.fromarray(cropped), config="--psm 7"))  # psm 7 = one-line text
#     # print("=" * 100)
#
#     angle = area["rotated"][-1]
#
#     image_center = tuple([1000, 0])
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rotated = cv2.warpAffine(cropped, rot_mat, (cropped.shape[1], cropped.shape[0]), flags=cv2.INTER_LINEAR)
#     contours, _ = cv2.findContours(rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     final_x, final_y, final_w, final_h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#     final = crop(rotated, (final_x, final_y), (final_x+final_w, final_y+final_h))
#
#     contours, hierarchy = cv2.findContours(final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#     hierarchy = hierarchy[0]
#
#     possible_chars = []
#     for relations in hierarchy:
#         print(relations)
#         if relations[2] != -1:  # Has child(ren)
#             child_i = relations[2]
#             possible_chars.append(contours[child_i])
#             while hierarchy[child_i][0] != -1:  # sibling
#                 child_i = hierarchy[child_i][0]
#                 possible_chars.append(contours[child_i])
#
#     for i, possible_char in enumerate(possible_chars):
#         x, y, w, h = cv2.boundingRect(possible_char)
#         possible_char_box = crop(final, (x, y), (x+w, y+h))
#         possible_char_box = cv2.GaussianBlur(possible_char_box, (7, 7), 0)
#         cv2.imshow(str(i), possible_char_box)
#         print("="*10)
#         print(pytesseract.image_to_boxes(Image.fromarray(possible_char_box), config="--psm 10"))
#         print("="*10)

    # boxes = map(lambda x: x.split(" "), pytesseract.image_to_boxes(Image.fromarray(final), config="--psm 7").split("\n")[:-1])
    # print(tuple(boxes))
    # final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # for char, x, y, w, h, _ in boxes:
    #     cv2.rectangle(final, (x, y), (x+w, y-h), (0, 255, 0))

# print("=" * 100)
# print(pytesseract.image_to_boxes(Image.fromarray(img), config="--psm 7"))  # psm 7 = one-line text
# print("=" * 100)

