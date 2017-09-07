# -*- default-directory: "~/Projects/mechlearn/fami/"; compile-command: "python -m nes.ntreg " -*-

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
fb_pix = np.array(emu_result["data"][-1]["framebuffer"])
plt.figure()
plt.imshow(fb_pix)
plt.savefig("out/tr-expected.png")

# let's check nametable registration of this frame
# should be two rectangular regions:
#  one at 0,0 in both FB and NTA, width = 256, height = 32
#  one at 0,32 in FB and ?,32 in NTA, width=256, height=240-32=208

print "Ready to go", state
#  Socket to talk to server
tsocket = context.socket(zmq.REQ)
tsocket.connect("tcp://127.0.0.1:5557")

print "Ask1"
tsocket.send_json(
    {"state": state,
     "data": ["tilemap", "tilepals", "tilemap_pixels",
              "palette", "tiles_to_patterns", "pals_to_colors",
              "color_tiles"]})
tile_result = tsocket.recv_json()

nta_pix = np.array(tile_result["tilemap_pixels"], dtype=np.float32)
# TODO: extend nta_pix by one nta rightwards wrapping and one nta
# downwards wrapping

plt.figure()
plt.imshow(nta_pix)
plt.savefig("out/tr-bg.png")
plt.close()
print "Good!"

ssocket = context.socket(zmq.REQ)
ssocket.connect("tcp://127.0.0.1:5558")

print "Ask1"
ssocket.send_json(
    {"state": state,
     "data": [
         "sprites",
         "color_sprites",
         "sprites_to_patterns",
         "palette",
         "pals_to_colors"
     ]})
sprite_result = ssocket.recv_json()

sprite_mask = np.zeroes(shape=fb_pix.shape, dtype=np.float32)

sprites = sprite_result["sprites"]

for s in sprites:
    x = s["x"]
    y = s["y"]
    sprite_mask[y:y + 8, x:x + 8, :] = 1.0

# NOW
# let's start from the top left pixel of the framebuffer
# and find the first pixel in the nametable which matches it???

# for fy in fb_pix.shape[0]:
#     nta_match_lines = []
#     finished_nta_match_lines = []
#     for fx in fb_pix.shape[1]:
#         # find bits of nta matching seed "matches"
#         # right-extend any nta-match-lines touching fx,fy if their right neighbor is in matches, end other nta-match-lines and create new nta-match-lines for those
#         seed = fb_pix[fy, fx]
#         matches = nta_pix.where(nta_pix == seed).zip()
#         if len(nta_match_lines) == 0:
#             pass
#         else:
#             for l in nta_match_lines:
#                 # try to extend; if it's finished
#                 pass

#         matches = matches - extended
#         nta_match_lines = nta_match_lines - finished + new_ones
#         print "matches:", len(matches)
#     finished_nta_match_lines += nta_match_lines

# now extend each match as much as possible
