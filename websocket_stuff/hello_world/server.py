from tornado import web, websocket, ioloop


class HelloWorldHandler(websocket.WebSocketHandler):
    def open(self):
        print("opened")

    def on_message(self, message):
        print(message, type(message))
        self.write_message("Hello, World!")

    def on_close(self):
        print("closed")


def main():
    app = web.Application([
        (u"/", HelloWorldHandler)
    ])
    app.listen(8888)
    ioloop.IOLoop.current().start()  # creates or gets IOLoop belonging to thread. IOLoop should work with asyncio.


if __name__ == "__main__":
    main()
