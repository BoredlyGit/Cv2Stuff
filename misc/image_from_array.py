import cv2
import numpy as np

my_img = [
    [
        [250, 250, 250, 0, 0, 0] * 100,
        [250, 250, 250, 0, 0, 0] * 100,
        [250, 250, 250, 0, 0, 0] * 100,
        [0, 0, 0, 0, 0, 0]*100,
        [0, 0, 0, 0, 0, 0] * 100,
        [0, 0, 0, 0, 0, 0] * 100,
    ]*100,
    [
        [0, 0, 0, 250, 250, 250] * 100,
        [0, 0, 0, 250, 250, 250] * 100,
        [0, 0, 0, 250, 250, 250] * 100,
        [0, 0, 0, 0, 0, 0] * 100,
        [0, 0, 0, 0, 0, 0] * 100,
        [0, 0, 0, 0, 0, 0] * 100,
    ]*100,
    [
        [100, 100, 100, 0, 0, 0] * 100,
        [100, 100, 100, 0, 0, 0] * 100,
        [100, 100, 100, 0, 0, 0] * 100,
        [255, 255, 255, 0, 0, 0] * 100,
        [255, 255, 255, 0, 0, 0] * 100,
        [255, 255, 255, 0, 0, 0] * 100,
    ] * 100,
]

my_img = cv2.merge(np.array(my_img, dtype=np.uint8))
print(my_img, type(my_img), my_img.shape)

cv2.imshow("test", my_img)
cv2.waitKey(0)