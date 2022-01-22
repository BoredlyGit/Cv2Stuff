import cv2
from cv2 import dnn
import numpy as np
# https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
# https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# https://github.com/argman/EAST/blob/master/eval.py


def nearest_mult_of_32(num):
    return round(num/32)*32 if round(num/32)*32 != 0 else 32


img = cv2.imread("images/test_2.jpg")
min_score = 0.85

# text_detector = dnn.TextRecognitionModel()

east = dnn.readNet("frozen_east_text_detection.pb")  # source: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
print(east.getLayerNames())
output_layers = [
    "feature_fusion/Conv_7/Sigmoid",  # Probability of a region containing any text
    "feature_fusion/concat_3",  # Feature map of image ("geometry")
]

h = 480
w = 640
img = cv2.resize(img, (w, h))
mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array([89, 103, 49]), np.array([135, 255, 204]))
mask = np.bitwise_not(mask)  # EAST only works correctly with white on black on binary images for some reason

blob = dnn.blobFromImage(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), scalefactor=1, swapRB=True, crop=False)  # mean subtraction (subtracts average color from all pixels in image)

east.setInput(blob)
scores, geometry = east.forward(outBlobNames=output_layers)  # see ln 19 - will return the outputs of those layers
# each are in a 1-element array in case of batching i guess
scores = scores[0][0]
geometry = geometry[0]

boxes = []
box_scores = []
box_dst_up, box_dst_down, box_dst_right, box_dst_left, angles = geometry

print(scores.shape)
for y in range(0, scores.shape[0]):
    for x in range(0, scores.shape[1]):
        score = scores[y][x]
        if score < min_score:
            continue

        pos_x, pos_y = (x*4, y*4)
        # TODO: maybe deal with rotation - output are rotated rectangles. current impl results in some odd bounding
        #  boxes, especially as angle approaches a multiple of 90
        angle = angles[y][x]

        """
        Image coordinate system - Origin (0, 0) is top left, right is +x, down is +y
        
        Geometry layer output - Outputs 5 arrays - 4 of distances and 1 of angles. Based around a central point given 
        by the index (actual position on image is index*4), each distance is the extent of the rectangle in that 
        direction, and the angle is the rectangle's rotation. See https://stackoverflow.com/a/55603701 for a better 
        explanation
        """
        x1 = int(pos_x - box_dst_left[y][x])
        y1 = int(pos_y - box_dst_up[y][x])
        x2 = int(pos_x + box_dst_right[y][x])  # currently not used, but kept just in case
        y2 = int(pos_y + box_dst_down[y][x])  # currently not used, but kept just in case
        w = int(box_dst_left[y][x] + box_dst_right[y][x])
        h = int(box_dst_up[y][x] + box_dst_down[y][x])

        boxes.append((x1, y1, w, h))  # cv2 rectangle representation
        box_scores.append(score)

nms_indexes = dnn.NMSBoxes(boxes, box_scores, min_score, 0.1)
print("scores: ", box_scores)
print("boxes: ", nms_indexes)

for index in nms_indexes:
    x, y, w, h = boxes[index]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
    cv2.putText(img, str(box_scores[index]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))


cv2.imshow("", img)
cv2.imshow("mask", mask)
cv2.waitKey(0)
