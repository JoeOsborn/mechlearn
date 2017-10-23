import zmq
import fceulib
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

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

t = 1824
print "go to", t / 2, "out of", (len(all_inputs) / 2)
print "Ask0"
socket.send_json({"state": 0,
                  "inputs": all_inputs[:t],
                  "lastn": 1,
                  "data": ["framebuffer"]})
emu_result = socket.recv_json()
state = emu_result["states"][-1]
fb = emu_result["data"][-1]["framebuffer"]
npimg = np.array(fb, dtype=np.uint8)
cv2.imwrite("out/expected1.png",
            cv2.cvtColor(npimg, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(4, 4))
plt.imshow(npimg)
plt.savefig("out/expected2.png")
