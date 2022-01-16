import cv2
from cv2 import dnn
import numpy as np
# https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
# https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# https://github.com/argman/EAST/blob/master/eval.py
# !! https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/


def nearest_mult_of_32(num):
    return round(num/32)*32 if round(num/32)*32 != 0 else 32


img = cv2.imread("images/Capture2.PNG")

# text_detector = dnn.TextRecognitionModel()

east = dnn.readNet("frozen_east_text_detection.pb")  # source: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
print(east.getLayerNames())
output_layers = [
    "feature_fusion/Conv_7/Sigmoid",  # Probability of a region containing any text
    "feature_fusion/concat_3",  # Feature map of image ("geometry")
]

h = nearest_mult_of_32(img.shape[0])  # if too slow, divide by 4 orr smt to make smaller images
w = nearest_mult_of_32(img.shape[1])  # if too slow, divide by 4 orr smt to make smaller images
blob = dnn.blobFromImage(img, scalefactor=1, size=(w, h), swapRB=True, crop=False, )  # mean subtraction (subtracts average color from all pixels in image)

east.setInput(blob)
scores, geometry = east.forward(outBlobNames=output_layers)  # see ln 19


cv2.imshow("og", img)
cv2.waitKey(0)
