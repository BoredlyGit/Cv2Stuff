# !!IMPORTANT!! DON'T USE PIP - see https://github.com/sirfz/tesserocr#windows for installation
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level
import cv2
from PIL import Image
import cProfile
import pstats
import numpy as np
import json

tesseract_engine = PyTessBaseAPI(path="tessdata", psm=PSM.SPARSE_TEXT)

with open("hsv_ranges.json") as hsv_json:
    hsv_ranges = json.load(hsv_json)


def main():
    use_video = False
    img_path = "images/test_7.jpg"
    w, h = (720, 420)  # lower size if too slow

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        if use_video:
            _, img = cap.read()  # noqa
        else:
            img = cv2.imread(img_path)

        img = cv2.resize(img, (w, h))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  TODO: uncomment when done with grayscale tests
        img = cv2.GaussianBlur(img, (5, 5), 0)

        cv2.imshow("f", img)

        # blue_mask, red_mask = np.zeros(img.shape[:2], dtype="uint8"), np.zeros(img.shape[:2], dtype="uint8")
        # for hsv_range in hsv_ranges["blue_bumper"]:
        #     blue_mask = np.bitwise_or(blue_mask, cv2.inRange(img, np.array(hsv_range["min"]), np.array(hsv_range["max"])))
        # blue_mask = cv2.dilate(cv2.erode(blue_mask, (3, 3)), (3, 3))
        # for hsv_range in hsv_ranges["red_bumper"]:
        #     red_mask = np.bitwise_or(red_mask, cv2.inRange(img, np.array(hsv_range["min"]), np.array(hsv_range["max"])))
        # combined_mask = np.bitwise_or(blue_mask, red_mask)
        # cv2.imshow("red", red_mask)
        # cv2.imshow("blue", blue_mask)
        # cv2.imshow("mask", combined_mask)
        #
        # canny = cv2.Canny(img, 250, 450)
        # cv2.imshow("canny", canny)

        print("starting")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # TODO: This seems to be the most accurate way of doing text recogniotion - find text in reduced color gray and
        #  check positions against color in image (color of center point of rect) - no need for contours?
        # color reduction
        div = 128
        img = img // div * div + div // 2
        # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(img)
        tesseract_engine.SetImage(Image.fromarray(img))
        # tesseract.Recognize()
        components = tesseract_engine.GetComponentImages(RIL.WORD, True)

        for component in components:
            print(type(component), component)
            rect = component[1]
            cv2.rectangle(img, (rect["x"], rect["y"]), (rect["x"] + rect["w"], rect["y"] + rect["h"]),
                          color=(255, 255, 255))

        cv2.imshow("img", img)
        if use_video:
            if cv2.waitKey(1) in (27, 113):  # 27 = esc, 113 = q
                break
        else:
            cv2.waitKey(0)
            break

        # img = cv2.resize(img, (w, h))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img = cv2.GaussianBlur(img, (3, 3), 0)
        #

        # # maybe consider image_to_data? - could be useful but rn just add complexity for no real benefit
        # # psm 11 = sparse text
        # boxes = pytesseract.image_to_boxes(Image.fromarray(combined_mask), config="--psm 11", output_type=pytesseract.Output.DICT)
        #
        # if boxes:  # tesseract returns empty dict if no text
        #     for i in range(len(boxes["left"])):
        #         x1, y1, x2, y2 = boxes["left"][i], boxes["top"][i], boxes["right"][i], boxes["bottom"][i]
        #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))


if __name__ == "__main__":
    cProfile.run('main()', filename="tessocr_profiler_out")
    stats = pstats.Stats("tessocr_profiler_out")
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()

    # results = (iterate_level(tesseract.GetIterator(), RIL.SYMBOL))
    # print(results)
    # for result in results:
    #     print(result.GetUTF8Text(RIL.SYMBOL))
    #     print(result.Confidence(RIL.SYMBOL))

    # # words = tesseract.GetWords()  # supposed to return list, but returns zip instead. Also crashes when attempting to access zip items lmao
    # for component in components:
    #     print(type(component), component)
        # component[0].show()

    # print(tesseract.AllWordConfidences())
    # for word in words:
    #     print(word)
    # print(tesseract.GetUTF8Text())

