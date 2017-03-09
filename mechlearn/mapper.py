import fceulib
import ppu_dump
#import tracking
import sys
import numpy as np
from PIL import Image

if __name__ == "__main__":
    rom = sys.argv[1]
    movie = sys.argv[2]
    start_t = int(sys.argv[3])
    outputname = sys.argv[4]
    emu = fceulib.runGame(rom)
    inputs = fceulib.readInputs(movie)
    for i in inputs[:start_t]:
        emu.stepFull(i, 0x0)
    ep_data = ppu_dump.ppu_output(emu,
                                  inputs[start_t:],
                                  bg_data=False,
                                  scrolling=True,
                                  sprite_data=False,
                                  colorized_tiles=True,
                                  display=False)
    posx, posy = 0, 0
    nt_total = {}
    for t, (sx, sy) in ep_data["screen_motion"].items():
        nt = ep_data["nametables"][t]
        posx += sx
        posy += sy
        print "t", t, "scroll by", sx, sy, posx, posy
        tilex = posx / 8
        tiley = posy / 8
        for x in range(0, 32):
            for y in range(0, 30):
                nt_total[(y+tiley, x+tilex)] = nt[y, x]
    minx = min(nt_total.keys().map(lambda x, y: x))
    maxx = max(nt_total.keys().map(lambda x, y: x))+1
    miny = min(nt_total.keys().map(lambda x, y: y))
    maxy = max(nt_total.keys().map(lambda x, y: y))+1
    nt_result = np.zeros(shape=(maxy-miny, maxx-minx))
    for x in range(minx, maxx):
        for y in range(miny, maxy):
            nt_result[miny+y, minx+x] = nt_total[(y, x)]
    colorized = ep_data["tile2colorized"]
    out_image = np.zeros(shape=((maxy-miny)*8, (maxx-minx)*8, 4))
    for x in range(0, maxx-minx):
        for y in range(0, maxy-miny):
            out_image[y*8, x*8, :] = colorized[nt_result[y, x]]
    outImg = Image.fromBytes("RGBA",
                             (maxx-minx, maxy-miny),
                             str(bytearray(out_image)))
    outImg.save(outputname+".png")
