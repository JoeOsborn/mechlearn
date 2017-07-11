import zmq
import fceulib
from fceulib import VectorBytes
import numpy as np
from fceu_help import *


def send_msg(socket, extra_data, A, flags=0, copy=True, track=False):
    md = dict(
        _dtype=str(A.dtype),
        _shape=A.shape,
    )
    md.update(extra_data)
    socket.send_json(md, flags | zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


# 'videos/happylee4-smb-warpless.fm2')
inputVec = fceulib.readInputs('Illustrative.fm2')

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

send_msg(socket, {"state": 0}, np.hstack(
    (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
     np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[:20, :])
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)

send_msg(socket, {"state": 0}, np.hstack(
    (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
     np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[:20, :])
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)


send_msg(socket, {"state": 18}, np.hstack(
    (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
     np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[18:20, :])
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)


send_msg(socket, {"state": 18}, np.hstack(
    (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
     np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[18:25, :])
result = socket.recv_json()
assert result["states"][-1] == 25, str(result)
