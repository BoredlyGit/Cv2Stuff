from tornado import web, websocket, ioloop
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
        print(message, type(message))
        # jpg does lossy compression,payload size is too large without
        jpg_frame = cv2.imencode(".jpg", self.cap.read()[1])[1]
        self.write_message(b64encode(jpg_frame))

    def on_close(self):
        print("closed")


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    app = web.Application([
        ("/", RawVideoHandler, {"cap": cap}),
    ])
    app.listen(8868)
    ioloop.IOLoop.current().start()  # creates or gets IOLoop belonging to thread. IOLoop should work with asyncio.


if __name__ == "__main__":
    main()
