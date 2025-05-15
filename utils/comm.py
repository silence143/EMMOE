import json
import socket

class Communicator_HLP:
    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('127.0.0.1', port))
        self.s.listen(1)
        print("Server is listening...")
        self.conn, self.addr = self.s.accept()
        print('Connected by', self.addr)

    def receive_env_images(self):
        data = self.conn.recv(1024)
        if data:
            return json.loads(data.decode())
        return None

    def send_subtask(self, subtask):
        self.conn.sendall(subtask.encode())

    def receive_feedback(self):
        data = self.conn.recv(1024)
        print(data.decode())
        self.conn.sendall("feedback received!".encode())
        return json.loads(data.decode())

    def close_connection(self):
        self.conn.close()
        self.s.close()

class Communicator_LLE:
    def __init__(self, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', port))

    def send_env_images(self, images):
        message = json.dumps(images)
        self.s.sendall(message.encode('utf-8'))

    def receive_subtask(self):
        subtask = self.s.recv(1024).decode()
        return subtask

    def send_feedback(self, feedback, signal):
        message = json.dumps([feedback, signal])
        self.s.sendall(message.encode('utf-8'))
        self.s.recv(1024).decode()

    def close_connection(self):
        self.s.close()
