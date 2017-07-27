import zmq
import fceulib
import numpy as np
import fceulib
import matplotlib.pyplot as plt

inputVec = fceulib.readInputs('Illustrative.fm2')
all_inputs = np.ravel(
    np.hstack((np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
               np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))).tolist()

context = zmq.Context()

print "Get started"
fceusock = context.socket(zmq.REQ)
fceusock.connect("tcp://127.0.0.1:5555")
t = 1200
print "Ask0"
fceusock.send_json({"state": 0, "inputs": all_inputs[:t], "lastn": 1})
state = fceusock.recv_json()["states"][-1]

print "Ready to go", state
#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5557")

print "Ask1"
socket.send_json(
    {"state": state,
     "data": ["tilemap", "tilepals", "tilemap_pixels",
              "palette", "tiles_to_patterns", "pals_to_colors",
              "color_tiles"]})
result = socket.recv_json()
print result["palette"]
print result["pals_to_colors"]
print len(result["color_tiles"])
plt.imshow(np.array(result["tilemap_pixels"]))
plt.savefig("out/expected.png")
for ci, cv in enumerate(result["color_tiles"].values()):
    plt.imshow(np.array(cv, dtype=np.uint8))
    plt.savefig("out/t" + str(ci) + ".png")
print "Good!"
