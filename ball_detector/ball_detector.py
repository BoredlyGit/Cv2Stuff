import cv2
import numpy
import numpy as np
import json
import pprint
import json
import utils

# Suggestion: record and return data in json
# Suggestion: try to detect blobs of balls somehow?


def detect_balls(img, profile, all_contours=False):
    if profile["img_resize"]:
        img = cv2.resize(img, profile["img_resize"])

    img = cv2.GaussianBlur(img, (3, 3), 0)
    # No grayscale cause using color masks

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # doesn't accept tuples for some reason.
    mask = np.zeros(img.shape[:2], dtype="uint8")
    # print(mask.shape)
    for i, hsv_range in enumerate(profile["hsv_ranges"]):
        mask = np.bitwise_or(mask, cv2.inRange(img, np.array(hsv_range["min"]), np.array(hsv_range["max"])))
        cv2.imshow(f"mask{i}", mask)

    # try to detect edges of overlapping balls - find edges which can be subtracted from the mask to create separations
    if profile["subtract_canny"]:
        _, ball_edges = cv2.threshold((cv2.GaussianBlur(cv2.Canny(img, 240, 250), (3, 3), 0)), 10, 255, cv2.THRESH_BINARY)
        # print(ball_edges)
        cv2.imshow("edges", ball_edges)
        ball_edges = numpy.bitwise_and(mask, ball_edges)
        mask = mask - ball_edges
    cv2.imshow("final mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not all_contours:
        # check size and shape, (see variables above)
        contours = [contour for contour in contours
                    if cv2.contourArea(contour) > profile["ball_min_area"]
                    and (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2 - cv2.contourArea(contour)
                         < (3.14 * cv2.minEnclosingCircle(contour)[1] ** 2) * (1 - profile["ball_min_coverage"]))]

    return list(map(cv2.minEnclosingCircle, contours)), contours


def circles_to_json(circles, frame_dimensions, cam_focal_len):
    ret = []
    for circle in circles:
        pos_x, pos_y = int(circle[0][0]), int(circle[0][1])
        pos_x = pos_x - frame_dimensions[0]/2
        pos_y = frame_dimensions[1]/2 - pos_y
        ret.append({"position": (pos_x, pos_y),
                    "radius": int(circle[1]),
                    "distance": utils.distance_to_object(cam_focal_len, 24, int(circle[1])*2)})
    return json.dumps(ret)


def main():
    img_path = "images/blue_ball.PNG"
    img_path = img_path.lower()

    font_kwargs = {"fontFace": cv2.FONT_HERSHEY_DUPLEX, "fontScale": 0.5}
    use_video = True
    video_profile = "images/blue_ball.PNG"
    rotate_video = 0
    cam_focal_length = 655  # my laptop camera's

    with open("profiles.json") as img_profiles:
        img_profiles = json.load(img_profiles)
    img_profile = img_profiles["default"]

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        img_profile.update(img_profiles.get(video_profile.lower(), {}))
        print(f"Using profile: {img_profile['name']}\n{pprint.pformat(img_profile)}")

        while True:
            frame_ready = False
            og_img = None
            while not frame_ready:
                frame_ready, og_img = cap.read()

            if rotate_video:  # in case video is upside down or something
                h, w = og_img.shape[:2]
                center = (w//2, h//2)
                og_img = cv2.warpAffine(og_img, cv2.getRotationMatrix2D(center, rotate_video, 1), (w, h))

            circles, contours = detect_balls(og_img.copy(), img_profile, all_contours=False)
            cv2.drawContours(og_img, contours, -1, (255, 255, 255), 2)
            circles_data = json.loads(circles_to_json(circles, (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cam_focal_length))

            for i, circle in enumerate(circles):
                circle_center = (int(circle[0][0]), int(circle[0][1]))
                circle_radius = int(circle[1])
                contour = contours[i]
                b_rect_x, b_rect_y, b_rect_w, b_rect_h = cv2.boundingRect(contour)

                text = f"""pos: {circles_data[i]['position']} | distance: {round(circles_data[i]['distance'], 2)} | radius: {circles_data[i]['radius']}"""
                cv2.circle(og_img, circle_center, circle_radius, (255, 0, 0), 2)
                cv2.rectangle(og_img, (b_rect_x, b_rect_y), (b_rect_x+b_rect_w, b_rect_y+b_rect_h), (255, 255, 255))
                utils.draw_text_box(og_img, text, circle_center, font_kwargs)

                # focal_len = utils.get_focal_length(100, circle_width_irl, b_rect_w)

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
        circles_data = json.loads(circles_to_json(circles, og_img.shape[:2], cam_focal_length))

        for i, circle in enumerate(circles):
            circle_center = (int(circle[0][0]), int(circle[0][1]))
            circle_radius = int(circle[1])
            contour = contours[i]
            b_rect_x, b_rect_y, b_rect_w, b_rect_h = cv2.boundingRect(contour)

            text = f"pos: {circles_data[i]['position']} | radius: {circles_data[i]['radius']}"
            cv2.circle(og_img, circle_center, circle_radius, (255, 0, 0), 2)
            cv2.rectangle(og_img, (b_rect_x, b_rect_y), (b_rect_x + b_rect_w, b_rect_y + b_rect_h), (255, 255, 255))
            utils.draw_text_box(og_img, text, circle_center, font_kwargs)

        cv2.imshow("processed", og_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
