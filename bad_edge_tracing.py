import cv2
import numpy as np
from progress_bar import ProgressBar


def trace_img(img, threshold=3, cvt_grayscale=True, clean_noise=True):

    """
    lower thresholds (1-3) work better on drawings, and higher on irl images (3-7)

    In order to be considered a border, a pixel must have a difference of color greater than the threshold to ALL of its
    neighbors (For some reason doing this with just ONE of its neighbors results in weird artifacts).

    cvt_grayscale is recommended, but can cause some borders to be overlooked.

    clean_noise removes "border" pixels that are not surrounded by other border pixels
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)  # helps reduce noise
    p_bar = ProgressBar(end=img.shape[0] - 1)
    print("Detecting edges...")

    if cvt_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        traced = np.zeros(img.shape, dtype=np.uint8)

        for r_num, row in enumerate(img):
            for p_num, pixel in enumerate(row):
                try:
                    if (
                            abs(pixel - row[p_num + 1]) > threshold and
                            abs(pixel - row[p_num - 1]) > threshold and
                            abs(pixel - img[r_num + 1][p_num]) > threshold and
                            abs(pixel - img[r_num - 1][p_num]) > threshold):
                        # print(max(pixel - row[p_num + 1]))
                        traced[r_num][p_num] = 255
                except IndexError:
                    pass
            if r_num % 5 == 0 or r_num == p_bar.end:
                p_bar.progress = r_num

    else:
        print(img.shape)
        traced = np.zeros(img.shape[:2], dtype=np.uint8)

        for r_num, row in enumerate(img):
            for p_num, pixel in enumerate(row):
                try:
                    if (
                            max(abs(pixel - row[p_num + 1])) > threshold and
                            max(abs(pixel - row[p_num - 1])) > threshold and
                            max(abs(pixel - img[r_num + 1][p_num])) > threshold and
                            max(abs(pixel - img[r_num - 1][p_num])) > threshold):
                        # print(max(pixel - row[p_num + 1]))
                        traced[r_num][p_num] = 255
                except IndexError:
                    pass
            if r_num % 100 == 0 or r_num == p_bar.end:
                p_bar.progress = r_num

    if clean_noise:
        noise_bar = ProgressBar(end=img.shape[0] - 1)
        print("Cleaning noise...")

        for r_num, row in enumerate(traced):
            for p_num, pixel in enumerate(row):
                if pixel == 255:
                    if (
                            row[p_num + 1] == 0 and
                            row[p_num - 1] == 0 and
                            traced[r_num + 1][p_num] == 0 and
                            traced[r_num - 1][p_num] == 0 #and
                            # traced[r_num + 1][p_num + 1] == 0 and
                            # traced[r_num + 1][p_num - 1] == 0 and
                            # traced[r_num - 1][p_num + 1] == 0 and
                            # traced[r_num - 1][p_num - 1] == 0
                    ):
                        traced[r_num][p_num] = 0
            if r_num % 5 == 0 or r_num == noise_bar.end:
                noise_bar.progress = r_num

    return traced


image = cv2.imread("images/test_2.jpg")
print(image)

#
# for i in (2, 5, 10, 15):
#     for g in (True, False):
#         traced_image = trace_img(image, i, g)
#         cv2.imwrite(f"./outputs/test_3/output_test_3_thresh={i}_gs={g}.jpg", traced_image)

cv2.imwrite("traced.jpg", trace_img(image, cvt_grayscale=True, threshold=2, clean_noise=True))
cv2.waitKey(0)
