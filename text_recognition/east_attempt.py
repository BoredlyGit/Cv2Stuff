import cProfile
import pstats
import cv2
from cv2 import dnn
import numpy as np
# https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
# https://learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/
# https://github.com/argman/EAST/blob/master/eval.py


def nearest_mult_of_32(num):
    return round(num/32)*32 if round(num/32)*32 != 0 else 32


# Load once here for speed
east = dnn.readNet("frozen_east_text_detection.pb")  # source: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
print(east.getLayerNames())
output_layers = [
    "feature_fusion/Conv_7/Sigmoid",  # Probability of a region containing any text
    "feature_fusion/concat_3",  # Feature map of image ("geometry")
]


def detect_text(img, min_score):
    # TODO!!: combine with color masking using red/blue contours, ref_pt, and pointPolygonTest()
    # TODO: gotta go fast(er)
    # SUGGESTION: crop to bottom of image cause that's where relevant bumpers prob are? - could increase speed too
    h = 256
    w = 320
    img = cv2.GaussianBlur(cv2.resize(img, (w, h)), (3, 3), 0)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array([0, 0, 150]), np.array([179, 75, 255]))
    # mask = np.bitwise_not(mask)  # EAST only works correctly with white on black on binary images for some reason
    cv2.imshow("mask", mask)

    blob = dnn.blobFromImage(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), scalefactor=1, swapRB=True, crop=False)  # mean subtraction (subtracts average color from all pixels in image)

    east.setInput(blob)
    scores, geometry = east.forward(outBlobNames=output_layers)  # see ln 19 - will return the outputs of those layers (this is kinda slow - ~0.3s per call?)
    # each are in a 1-element array in case of batching i guess
    scores = scores[0][0]
    geometry = geometry[0]

    boxes = []
    box_scores = []
    box_dst_up, box_dst_down, box_dst_right, box_dst_left, angles = geometry

    # Optimize by using numpy to filter out good scores and get indexes of high scores
    good_y, good_x = (scores >= min_score).nonzero()  # noqa
    print(good_x, good_y)

    print(scores.shape)
    for i, y in enumerate(good_y):
        x = good_x[i]
        print(f"row: {y} col: {x}")
        score = scores[y][x]

        pos_x, pos_y = (x*4, y*4)
        # Suggestion: masybe deal with rotation - output are rotated rectangles. current impl results in some odd
        #  bounding boxes, especially as angle approaches a multiple of 90
        # TODO: are bounding boxes even necessary if only checking if point where EAST sees text is in a contour? - this
        #   might also remove the need form NMSBoxes, maybe more points = more likely bumper??
        # angle = angles[y][x]

        """
        Image coordinate system - Origin (0, 0) is top left, right is +x, down is +y
        
        Geometry layer output - Outputs 5 arrays - 4 of distances and 1 of angles. Based around a central point given 
        by the index (actual position on image is index*4), each distance is the extent of the rectangle in that 
        direction, and the angle is the rectangle's rotation. See https://stackoverflow.com/a/55603701 for a better 
        explanation
        """
        x1 = int(pos_x - box_dst_left[y][x])
        y1 = int(pos_y - box_dst_up[y][x])
        # x2 = int(pos_x + box_dst_right[y][x])  # currently not used, but kept just in case
        # y2 = int(pos_y + box_dst_down[y][x])  # currently not used, but kept just in case
        w = int(box_dst_left[y][x] + box_dst_right[y][x])
        h = int(box_dst_up[y][x] + box_dst_down[y][x])

        # cv2 rectangle representation + EAST coordinate point (see above docstring) (used to check if text is within
        # contour)
        boxes.append(((x1, y1, w, h), (pos_x, pos_y)))
        box_scores.append(score)

    nms_indexes = dnn.NMSBoxes(list(box[0] for box in boxes), box_scores, min_score, 0.05)
    print("scores: ", box_scores)
    print("boxes: ", nms_indexes)

    ret_scores = []
    ret_boxes = []
    for index in nms_indexes:
        ret_boxes.append((boxes[index]))
        ret_scores.append(box_scores[index])

    return ret_boxes, ret_scores


def main():
    min_score = 0.85
    use_video = False
    img_path = "images/test_3.jpg"

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        img = cv2.imread(img_path)

    for i in range(10):  # for use with testing video w/ profiler
    # while True:
        if use_video:
            frame_ready, img = cap.read()
            if not frame_ready:
                continue

        h = 256
        w = 320
        img = cv2.resize(img, (w, h))
        boxes, scores = detect_text(img, min_score)

        for i, box in enumerate(boxes):
            print(i)
            rect, ref_pt = box
            x, y, w, h = rect

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
            cv2.circle(img, ref_pt, 10, (255, 0, 255))
            cv2.putText(img, str(scores[i]), (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

        cv2.imshow("final", img)

        if use_video:
            if cv2.waitKey(1) in (27, 113):  # 27 = esc, 113 = q
                break
        else:
            cv2.waitKey(0)
            break


if __name__ == "__main__":
    cProfile.run('main()', filename="profiler_out")
    stats = pstats.Stats("profiler_out")
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    # see profiler_results.txt
