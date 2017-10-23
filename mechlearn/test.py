import zmq
import fceulib
from fceulib import VectorBytes
import numpy as np
from fceu_help import *

inputVec = fceulib.readInputs('Illustrative.fm2')

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

socket.send_json({"state": 0,
                  "data": ["nta", "pal", "ppu", "oam", "chr"],
                  "inputs": np.hstack(
                      (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
                       np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[:20, :],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)
assert len(result["data"]) == 2, str(result)

socket.send_json({"state": 0,
                  "inputs": np.hstack(
                      (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
                       np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[:20, :],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)

socket.send_json({"state": 18,
                  "inputs": np.hstack(
                      (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
                       np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[18:20, :],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)

socket.send_json({"state": 18,
                  "inputs": np.hstack(
                      (np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
                       np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))[18:25, :],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 25, str(result)
