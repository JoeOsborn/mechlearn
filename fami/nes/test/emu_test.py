import zmq
import fceulib
import numpy as np

inputVec = fceulib.readInputs('Illustrative.fm2')

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

all_inputs = np.ravel(
    np.hstack((np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
               np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))).tolist()

socket.send_json({"state": 0,
                  "data": ["nta", "pal", "ppu", "oam", "chr"],
                  "inputs": all_inputs[:20 * 2],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)
assert len(result["data"]) == 2, str(result)

socket.send_json({"state": 0,
                  "inputs": all_inputs[:20 * 2],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)

socket.send_json({"state": 18,
                  "inputs": all_inputs[18 * 2:20 * 2],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 20, str(result)

socket.send_json({"state": 18,
                  "inputs": all_inputs[18 * 2:25 * 2],
                  "data": ["nta", "framebuffer"],
                  "lastn": 2})
result = socket.recv_json()
assert result["states"][-1] == 25, str(result)
print np.array(result["data"][-1]["nta"])
oldfb = result["data"][-1]["framebuffer"]

socket.send_json({"state": 5,
                  "inputs": [],
                  "lastn": 1,
                  "data": ["framebuffer"]})
result = socket.recv_json()
assert result["states"][-1] == 5, str(result)
newfb = result["data"][-1]["framebuffer"]
assert oldfb != newfb
