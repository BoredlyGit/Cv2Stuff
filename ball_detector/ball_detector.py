import cv2
import numpy
import numpy as np
import tkinter
import pprint
import json
import utils
import gui_elements
# Suggestion: try to detect blobs of balls somehow?
# TODO: Other bot detection could possibly be done by detecting the bumpers - some combination of text, color, and shape
# todo: better red detection - maybe invert and look for cyan??


def detect_balls(img, profile, all_contours=False):
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
                    and (cv2.contourArea(contour)/(3.14 * cv2.minEnclosingCircle(contour)[1] ** 2)) > profile["ball_min_coverage"]]

    return list(map(cv2.minEnclosingCircle, contours)), contours


def circles_to_data(circles, frame_dimensions, cam_focal_len, use_metric=False, to_json=False):
    ret = []
    for circle in circles:
        pos_x, pos_y = int(circle[0][0]), int(circle[0][1])
        pos_x = pos_x - frame_dimensions[0]/2
        pos_y = frame_dimensions[1]/2 - pos_y
        diameter = int(circle[1])*2
        diameter = 1 if diameter == 0 else diameter  # prevent ZeroDivisions
        ret.append({"position": (pos_x, pos_y),
                    "radius": int(circle[1]),
                    "distance": utils.distance_to_object(cam_focal_len, 24, diameter) if use_metric else (utils.distance_to_object(cam_focal_len, 24, diameter)/30.48)})
    return json.dumps(ret) if to_json else ret


def main():
    use_gui = True
    profile_name = "all_balls"
    font_kwargs = {"fontFace": cv2.FONT_HERSHEY_DUPLEX, "fontScale": 0.5}

    img_path = "images/blue_ball.PNG"
    img_path = img_path.lower()

    use_video = True
    rotate_video = 0
    cam_focal_length = 655  # my laptop camera's - see utils.get_focal_length()

    with open("profiles.json") as profiles:
        profiles = json.load(profiles)
    profile = profiles["default"]
    profile.update(profiles.get(profile_name.lower(), {}))
    print(f"Using profile: {profile['name']}\n{pprint.pformat(profile)}")

    if use_video:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        og_img = cv2.imread(img_path)

    if use_gui:
        gui = gui_elements.OptionsFrame(init_options=profile)
        gui.pack()
        profile = gui.options

    while True:
        if use_video:
            frame_ready = False
            og_img = None
            while not frame_ready:
                frame_ready, og_img = cap.read()

            if rotate_video:  # in case video is upside down or something
                h, w = og_img.shape[:2]
                center = (w//2, h//2)
                og_img = cv2.warpAffine(og_img, cv2.getRotationMatrix2D(center, rotate_video, 1), (w, h))

        if profile["img_resize"] is not None:
            try:
                og_img = cv2.resize(og_img, profile["img_resize"])
            except cv2.error:
                print(f"RESIZE TO {profile['img_resize']} FAILED, try dimensions more similar to each other")

        circles, contours = detect_balls(og_img.copy(), profile, all_contours=False)

        processed_img = og_img.copy()
        cv2.drawContours(processed_img, contours, -1, (255, 255, 255), 2)
        circles_data = circles_to_data(circles, processed_img.shape[:2], cam_focal_length, profile["use_metric"])

        for i, circle in enumerate(circles):
            circle_center = (int(circle[0][0]), int(circle[0][1]))
            circle_radius = int(circle[1])
            # contour = contours[i]
            # b_rect_x, b_rect_y, b_rect_w, b_rect_h = cv2.boundingRect(contour)

            text = f"""pos: {circles_data[i]['position']} | distance: {round(circles_data[i]['distance'], 2)} | radius: {circles_data[i]['radius']}"""
            cv2.circle(processed_img, circle_center, circle_radius, (0, 255, 0), 2)
            # cv2.rectangle(processed_img, (b_rect_x, b_rect_y), (b_rect_x+b_rect_w, b_rect_y+b_rect_h), (255, 255, 255))
            utils.draw_text_box(processed_img, text, circle_center, font_kwargs)
            cv2.line(processed_img, (processed_img.shape[0]//2, processed_img.shape[1]), circle_center, (0, 255, 0), thickness=2)

        cv2.imshow("processed", processed_img)

        if use_gui:
            gui.update()
            try:
                if gui.options != profile:
                    profile = gui.options
                    print(profile)
            except tkinter.TclError:
                exit(-1)

        try:
            key = cv2.waitKey(1)
            if key in (27, 113):  # 27 = esc, 113 = q
                print(key)
                break
        except KeyboardInterrupt:
            exit(-1)


if __name__ == "__main__":
    main()
