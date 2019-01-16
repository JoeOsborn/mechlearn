import remocon
import itertools
from PIL import Image
import os
import csv

# read playfile
fm2file = "lordtom_tompa-smb3-100.fm2"
info = remocon.read_fm2("../plays/" + fm2file)
romname = info["romFilename"][0] + ".nes"
controls_fm = info["player_controls"]
controls = list(map(remocon.moves_to_bytes, controls_fm))
pad = 10
for p in range(pad):
    for c in controls:
        c.insert(0, 0)
skip = 0
for ci in range(len(controls)):
    controls[ci] = controls[ci][skip:]
tmax = min(len(controls[0]), 700)
print("Load ../roms/{}".format(romname))
remo = remocon.Mesen(
    "../mesen/remocon/obj.x64/remocon",
    "../roms/" + romname)

os.makedirs("../data/{}/{}".format(romname, fm2file), exist_ok=True)

remo.step([controls[0][0:tmax - 1]], remocon.Infos())
frames = remo.step([controls[0][tmax - 1:tmax]], remocon.Infos(framebuffer=True))[0]
# for t, frame in enumerate(frames):
# fb = frame.framebuffer
Image.fromarray(frames[0].framebuffer).save("../data/{}/{}/t_{}.png".format(romname, fm2file, tmax - 1))

# def scroll_mode(tiles_by_pixel):
#     scrolls = {}
#     mode = None
#     mode_count = 0
#     for row in tiles_by_pixel:
#         for pix in row:
#             if pix.x_scroll != 0 or pix.y_scroll != 0:
#                 print(t, pix.x_scroll, pix.y_scroll)
#             xsc = int(pix.x_scroll)
#             ysc = int(pix.y_scroll)
#             key = (xsc, ysc)
#             scrolls[key] = scrolls.get(key, 0) + 1
#             if scrolls[key] > mode_count:
#                 mode = key
#                 mode_count = scrolls[key]
#     return mode


# tskip = 500 + (hash(fm2file) % 100)
# remo.step([controls[0][0:tskip - 1]], remocon.Infos())
# tsteps = [11, 5, 7, 19, 3, 15, 1, 13, 23]
# stepper = itertools.cycle(tsteps)
# t = tskip
# frame0 = remo.step(
#     [controls[0][tskip - 1:tskip]],
#     remocon.Infos(framebuffer=True,
#                   tiles_by_pixel=True))[0][0]
# fb0 = frame0.framebuffer
# scroll0 = scroll_mode(frame0.tiles_by_pixel)
# scrolls = [(t, 0, 0)]
# while t < tmax:
#     t0 = t
#     tstep = next(stepper)
#     t += tstep
#     if tstep > 1:
#         remo.step([controls[0][t0:t - 1]], remocon.Infos())
#     frame = remo.step([controls[0][t - 1:t]], remocon.Infos(framebuffer=True, tiles_by_pixel=True))[0][0]
#     fb = frame.framebuffer
#     scroll = scroll_mode(frame.tiles_by_pixel)

#     # write out fb for t
#     Image.fromarray(fb).save("../data/{}/{}/t_{}.png".format(romname, fm2file, t))
#     # add t, scrolldelta to csv
#     scrolls.append((t, scroll[0] - scroll0[0], scroll[1] - scroll0[1]))

#     fb0 = fb
#     scroll0 = scroll

# with open("../data/{}/{}/scrolls.csv".format(romname, fm2file), 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(scrolls)
