import cv2
from cv2 import dnn
import numpy as np
# https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
# https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# https://github.com/argman/EAST/blob/master/eval.py
# !! https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/


def nearest_mult_of_32(num):
    return round(num/32)*32 if round(num/32)*32 != 0 else 32


img = cv2.imread("images/Capture.PNG")

# text_detector = dnn.TextRecognitionModel()

east = dnn.readNet("frozen_east_text_detection.pb")  # source: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
print(east.getLayerNames())
output_layers = [
    "feature_fusion/Conv_7/Sigmoid",  # Probability of a region containing any text
    "feature_fusion/concat_3",  # Feature map of image ("geometry")
]

h = nearest_mult_of_32(img.shape[0])  # if too slow, divide by 4 or smt to make smaller images
w = nearest_mult_of_32(img.shape[1])  # if too slow, divide by 4 or smt to make smaller images
blob = dnn.blobFromImage(img, scalefactor=1, size=(w, h), swapRB=True, crop=False)  # mean subtraction (subtracts average color from all pixels in image)

east.setInput(blob)
scores, geometry = east.forward(outBlobNames=output_layers)  # see ln 19 - will return the outputs of those layers
# each are in a 1-element array in case of batching i guess
scores = scores[0][0]
geometry = geometry[0]

rects = []
rect_scores = []
box_dst_up, box_dst_down, box_dst_right, box_dst_left, angles = geometry

print(scores.shape)
for y in range(0, scores.shape[0]):
    for x in range(0, scores.shape[1]):
        score = scores[y][x]
        if score < 0.5:
            continue

        pos_x, pos_y = (x*4, y*4)
        # angle = angles[y][x]

        # image coordinate system: origin is top left, right is +x, down is +y
        x1 = int(pos_x - box_dst_left[y][x])
        y1 = int(pos_y - box_dst_up[y][x])
        x2 = int(pos_x + box_dst_right[y][x])
        y2 = int(pos_y + box_dst_down[y][x])
        w = int(box_dst_left[y][x] + box_dst_right[y][x])
        h = int(box_dst_up[y][x] + box_dst_down[y][x])

        rects.append((x1, y1, w, h))
        rect_scores.append(score)

boxes = dnn.NMSBoxes(rects, rect_scores, 0.5, 0.1)
print("scores: ", rect_scores)
print("rects: ", boxes)

for box in boxes:
    x, y, w, h = rects[box]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))


cv2.imshow("", img)
cv2.waitKey(0)
