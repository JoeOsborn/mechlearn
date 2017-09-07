import zmq
import fceulib
import numpy as np
import matplotlib.pyplot as plt

inputVec = fceulib.readInputs('Illustrative.fm2')
all_inputs = np.ravel(
    np.hstack((np.array(inputVec, dtype=np.uint8).reshape((-1, 1)),
               np.zeros(shape=(len(inputVec), 1), dtype=np.uint8)))).tolist()

context = zmq.Context()

print "Get started"
fceusock = context.socket(zmq.REQ)
fceusock.connect("tcp://127.0.0.1:5555")
t = 1824
print "go to", t / 2, "out of", (len(all_inputs) / 2)
print "Ask0"
fceusock.send_json({"state": 0,
                    "inputs": all_inputs[:t],
                    "lastn": 1,
                    "data": ["framebuffer"]})
emu_result = fceusock.recv_json()
state = emu_result["states"][-1]
fb = emu_result["data"][-1]["framebuffer"]
plt.figure()
plt.imshow(np.array(fb))
plt.savefig("out/expected.png")


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

# TODO: have some assertions to check actual vs expected

print result["palette"]
print result["pals_to_colors"]
print len(result["color_tiles"])
plt.figure()
plt.imshow(np.array(result["tilemap_pixels"]))
plt.savefig("out/bg.png")
plt.close()
for ci, cv in (result["color_tiles"].items()):
    for ci2, cv2 in cv.items():
        plt.figure()
        plt.imshow(np.array(cv2, dtype=np.float32))
        plt.savefig("out/t" + str(ci) + "_" + str(ci2) + ".png")
        plt.close()
print "Good!"
