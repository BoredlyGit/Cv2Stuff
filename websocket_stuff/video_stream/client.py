import numpy
import websockets
import asyncio
import cv2
from base64 import b64decode
import numpy as np
from io import BytesIO


async def main():
    async with websockets.connect("ws://localhost:8868/raw/") as ws:
        while True:
            print("f")
            await ws.send("frame")
            # cast to numpy array for cv2 - will not be required when move to js
            frame = numpy.asarray(bytearray(b64decode(await ws.recv())))
            print(type(frame), frame)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

asyncio.run(main())
