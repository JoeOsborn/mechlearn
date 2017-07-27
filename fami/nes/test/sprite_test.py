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
t = 1850
print "Ask0"
fceusock.send_json({"state": 0, "inputs": all_inputs[:t], "lastn": 1})
state = fceusock.recv_json()["states"][-1]

print "Ready to go", state
#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5558")

print "Ask1"
socket.send_json(
    {"state": state,
     "data": [
         "sprites",
         "color_sprites",
         "sprites_to_patterns",
         "palette",
         "pals_to_colors"
     ]})
result = socket.recv_json()

# TODO: have some assertions to check actual vs expected

print result["palette"]
print result["pals_to_colors"]
print len(result["color_sprites"])
print result["sprites"]
print result["color_sprites"]
for ci, cv in result["color_sprites"].items():
    for ci2, cv2 in cv.items():
        plt.figure()
        plt.imshow(np.array(cv2, dtype=np.float32))
        plt.savefig("out/s" + str(ci) + "_" + str(ci2) + ".png")
        plt.close()
print "Good!"
