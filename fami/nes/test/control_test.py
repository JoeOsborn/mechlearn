import zmq
import fceulib
import numpy as np
import fceulib

inputVec = fceulib.readInputs('Illustrative.fm2')
all_inputs = np.ravel(
    np.hstack((np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
               np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))).tolist()

context = zmq.Context()

print "Get started"
fceusock = context.socket(zmq.REQ)
fceusock.connect("tcp://127.0.0.1:5555")
print "Ask0",all_inputs[:600]
fceusock.send_json({"state":0, "inputs":all_inputs[:600], "lastn":1})
state = fceusock.recv_json()["states"][-1]

print "Ready to go",state
#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5556")

socket.send_json({"state": state,
                  "horizon": 20})
print "Ask1"
result = socket.recv_json()
assert result["has_control"] == True

print "Get paused"
fceusock.send_json({"state":state, "inputs":[1 << 3, 0]})
state_paused = fceusock.recv_json()["states"][-1]

print "ask2",state_paused
socket.send_json({"state": state_paused,
                  "horizon": 10})
result = socket.recv_json()
assert result["has_control"] == False
print "Good!"

