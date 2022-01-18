import tornado
from tornado import web, websocket, ioloop, iostream
import cv2
from base64 import b64encode
import asyncio
import numpy as np


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
    def on_message(self, message):
        if message == "frame":
            frame = self.cap.read()[1]
            frame = # TODO: Process
            self.send_frame(frame)
        elif message == "get_config":
            pass


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
