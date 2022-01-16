import websockets
import asyncio


async def hello_world():
    async with websockets.connect("ws://localhost:8888/") as ws:
        await ws.send("Hello World")
        reply = await ws.recv()
        print(reply)


asyncio.run(hello_world())
