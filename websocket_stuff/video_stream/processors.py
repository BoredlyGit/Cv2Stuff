class BaseProcessor:
    """
    Base class for all processors - They should subclass and overwrite __init__() and run(). In __init__, all
    NotImplementedError values should be replaced with actual ones.
    """
    def __init__(self):
        """
        self.name - The name to be used by this processor. All settings for this processor in config.json will be under
                    this key, and messages sent to and from control websockets will include this name to indicate that
                    this processor's settings are being changed. MUST be unique, "global" is a reserved name, to change
                    the server's settings or enable/disable processors.
        """
        self.name = NotImplementedError

    def run(self, frame):
        """
        Run the processor on the given cv2 img/np array. Should return a tuple of (to_draw, data).

        to_draw - Iterable of dictionaries of {"item": item, "kwargs": {**kwargs}}, where item is one of "line",
                  "rect", "circle", "text", "contours" and kwargs is the arguments to be passed to the corresponding cv2
                   function (ex: "rectangle" -> cv2.rect(), "contours" -> cv2.drawContours(), etc.).

                  Meant to replace processors directly drawing on images so that all "drawings" can be layered on the
                  final processed frame.

        data - a dict to be turned to json and sent to the client (via the data ws). (Can be empty)
        """
        raise NotImplementedError
