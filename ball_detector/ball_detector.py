import cv2
import numpy
import numpy as np
import json
import pprint

# TODO: record and return data
# TODO: try to detect blobs of balls somehow?
# Suggestion: use ellipses instead and check ratio?


def detect_balls(img, profile, all_contours=False):
    if profile["img_resize"]:
        img = cv2.resize(img, profile["img_resize"])

    img = cv2.GaussianBlur(img, (3, 3), 0)
    # No grayscale cause using color masks

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # doesnt accept tuples for some reason.
    mask = cv2.inRange(img, np.array(profile["ball_hsv_min"]), np.array(profile["ball_hsv_max"]))

    # try to detect edges of overlapping balls with Canny - it finds edges which can be subtracted from the mask
    ball_edges = cv2.GaussianBlur(cv2.Canny(img, 0, 250), (3, 3), 0)
    ball_edges = numpy.logical_and(mask, ball_edges)
    ball_edges.dtype = np.dtype("uint8")
    ball_edges = ball_edges * 255
    mask = mask - ball_edges

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not all_contours:
        # check size and shape, (see variables above)
        contours = [contour for contour in contours
                    if cv2.contourArea(contour) > profile["ball_min_area"]
                    and (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2 - cv2.contourArea(contour)
                         < (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2) * (1 - profile["ball_min_coverage"]))]

    return list(map(cv2.minEnclosingCircle, contours)), contours


def main():
    img_path = "images/test_1.jpg"
    img_path = img_path.lower()
    use_video = False
    camera_name = ""
    rotate_video = 0

    with open("profiles.json") as img_profiles:
        img_profiles = json.load(img_profiles)
    img_profile = img_profiles["default"]

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        img_profile.update(img_profiles.get(camera_name, {}))
        print(f"Using profile: {img_profile['name']}\n{pprint.pformat(img_profile)}")

        while True:
            frame_ready = False
            og_img = None
            while not frame_ready:
                frame_ready, og_img = cap.read()

            if rotate_video:
                h, w = og_img.shape[:2]
                print(w, h)
                center = (w//2, h//2)
                og_img = cv2.warpAffine(og_img, cv2.getRotationMatrix2D(center, rotate_video, 1), (w, h))

            circles, contours = detect_balls(og_img.copy(), img_profile)
            cv2.drawContours(og_img, contours, -1, (255, 255, 255), 2)
            for circle in circles:
                cv2.circle(og_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (255, 0, 0), 2)

            cv2.imshow("processed", og_img)
            key = cv2.waitKey(1)
            if key in (27, 113):  # 27 = esc, 113 = q
                print(key)
                break
    else:
        og_img = cv2.imread(img_path)
        img_profile.update(img_profiles.get(img_path, {}))
        print(f"Using profile: {img_profile['name']}\n{pprint.pformat(img_profile)}")

        circles, contours = detect_balls(og_img.copy(), img_profile)

        cv2.drawContours(og_img, contours, -1, (255, 255, 255), 2)
        for circle in circles:
            cv2.circle(og_img, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (255, 0, 0), 2)
        cv2.imshow("processed", og_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
