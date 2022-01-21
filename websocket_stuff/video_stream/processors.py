import json

import cv2

with open("config.json", "r+") as _config_file:
    _config = json.load(_config_file)


class BaseProcessor:
    """
    Base class for all processors - They should subclass and overwrite run(). Class variables set to NotImplementedError
    should be replaced with actual values.

    The configuration for the processor can be retrieved via self.config, which is a dictionary. If the processor lacks
    a configuration, it will be empty. Note that this is not the entire config file - just the part under the key of
    the processor's name

    Variables to replace:
    name[str] - The name to be used by this processor. All settings for this processor in config.json will be under
                this key, and messages sent to and from control websockets will include this name to indicate that
                this processor's settings are being changed. MUST be unique, "global" is a reserved name, to change
                the server's settings or enable/disable processors.
    """
    name = NotImplementedError

    def __init__(self):
        self.config = _config.get(self.name, {})

    def run(self, frame):
        """
        Run the processor on the given cv2 img/np array. Should return a tuple of (to_draw, data).

        to_draw - Iterable of dictionaries of {"item": item, "kwargs": {**kwargs}}, where item is one of "line",
                  "rect", "circle", "text", "contours" and kwargs is the arguments to be passed to the corresponding cv2
                   function (ex: "rectangle" -> cv2.rect(), "contours" -> cv2.drawContours(), etc.).

                  Meant to replace processors directly drawing on images so that all "drawings" can be layered on the
                  final processed frame. (This feels like a dumb hack but is the best solution I can think of to handle
                  multiple processors without doing output comparison or feeding the output of one into another)

        data - a dict to be turned to json and sent to the client (via the data ws). (Can be empty)
        """
        raise NotImplementedError


from ball_detector import ball_detector


class BallProcessor(BaseProcessor):
    name = "ball_detector"

    def run(self, frame):
        circles, contours = ball_detector.detect_balls(frame, profile=self.config)
        return (tuple({"item": "circle",
                       "kwargs": {
                           "center": circle[0],
                           "radius": circle[1],
                           "color": self.config["draw_color"]}} for circle in circles),

                ball_detector.circles_to_data(circles,
                                              frame_dimensions=frame.shape[:2],
                                              cam_focal_len=self.config["cam_focal_len"],
                                              use_metric=self.config["use_metric"]))
