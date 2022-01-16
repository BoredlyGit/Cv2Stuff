import cv2
from cv2 import dnn
import numpy as np


def nearest_mult_of_32(num):
    return round(num/32)*32 if round(num/32)*32 != 0 else 32


img = cv2.imread("images/Capture2.PNG")


east = dnn.readNet("EAST.pb")
output_layers = [
    "feature_fusion/Conv7/Sigmoid",  # Probability of a region containing any text
    "feature_fusion/concat_3",  # Feature map of image ("geometry")
]

h = nearest_mult_of_32(img.shape[0])
w = nearest_mult_of_32(img.shape[1])
blob = dnn.blobFromImage(img, scalefactor=1, size=(w, h), swapRB=True, crop=False, )  # mean subtraction (subtracts average color from all pixels in image)
print(blob.shape, blob)
east.setInput(blob)
scores, geometry = east.forward(output_layers)  # see ln 14
print(scores)

cv2.imshow("og", img)
cv2.waitKey(0)
