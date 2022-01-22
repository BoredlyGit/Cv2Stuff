import json
import tornado
from tornado import web, websocket, ioloop, iostream
import cv2
from base64 import b64encode
import asyncio
import numpy as np
from processors import BallProcessor
# Suggestion: Broadcast/waiter style impl, like in 1050-105?


class RawVideoHandler(websocket.WebSocketHandler):
    def initialize(self, cap):
        self.cap = cap  # noqa

    def open(self):
        print("opened")

    def on_message(self, message):
        self.send_frame(self.cap.read()[1])

    def send_frame(self, frame):
        # jpg does lossy compression,payload size is too large without
        jpg_frame = cv2.imencode(".jpg", frame)[1]
        try:
            self.write_message(b64encode(jpg_frame))
        except tornado.iostream.StreamClosedError:
            pass

    def on_close(self):
        print("closed")


class ProcessedVideoHandler(RawVideoHandler):
    # TODO: I have no idea if any of this works

    def initialize(self, cap):
        super().initialize(cap)
        self.enabled_processors = [BallProcessor()]  # TODO: allow enabling/disabling by name - this is just to get it working

    def on_message(self, message):
        draw_funcs = {
            "rect": cv2.rectangle,
            "circle": cv2.circle,
            "line": cv2.line,
            "contours": cv2.drawContours
        }

        if message == "frame":
            frame = self.cap.read()[1]
            to_draw = []
            data = {}
            for processor in self.enabled_processors:
                result = processor.run(frame)
                to_draw += result[0]
                data[processor.name] = result[1]

            for item in to_draw:
                print(item)
                frame = draw_funcs[item["item"]](frame, **item["kwargs"])
            self.send_frame(frame)

            self.write_message(json.dumps(data))


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    app = web.Application([
        ("/raw/", RawVideoHandler, {"cap": cap}),
        ("/processed/", ProcessedVideoHandler, {"cap": cap}),
    ])
    app.listen(8868)
    ioloop.IOLoop.current().start()  # creates or gets IOLoop belonging to thread. IOLoop should work with asyncio.


if __name__ == "__main__":
    main()
