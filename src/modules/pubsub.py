import zmq
import threading
import json
import time

class IPCPubSub:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.port = port
        self._publisher = None          # cached — only one bind per port

    def create_publisher(self):
        if self._publisher is None:
            self._publisher = IPCPublisher(self.context, self.port)
        return self._publisher          # always return the same instance

    def create_subscriber(self):
        return IPCSubscriber(self.context, self.port)

class IPCPublisher:
    def __init__(self, context, port):
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        time.sleep(0.2)

    def publish(self, topic, message):
        self.socket.send_string(f"{topic} {json.dumps(message)}")

class IPCSubscriber:
    def __init__(self, context, port):
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{port}")
        self.callbacks = {}

    def subscribe(self, topic, callback):
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.callbacks[topic] = callback

    def start(self):
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            try:
                msg = self.socket.recv_string()
                topic, payload = msg.split(" ", 1)
                data = json.loads(payload)
                if topic in self.callbacks:
                    self.callbacks[topic](data)
            except Exception as e:
                print(f"[IPC] Subscriber error: {e}")