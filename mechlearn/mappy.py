import fceulib
import json
import ppu_dump
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fceu_help import pointer_to_numpy
import scipy
import scipy.misc
import cv2
import pickle
from typing import List, Dict, Tuple, Sequence

# Type aliases
Rect = Tuple[int, int, int, int]
Inputs = Sequence[int]
Time = int
Tile = Tuple[int, int, int]
RowCol = Tuple[int, int]
ColRow = Tuple[int, int]
Room = Dict[RowCol, Dict[Time, Tile]]
SpriteData = Tuple[int, ColRow,  int]
Track = Dict[Time, Tuple[ColRow, Bbox, SpriteData]]


class Mappy:
    def __init__(self,
                 rom: str, movie: str,
                 scroll_area: Rect, start_t: int, end_t: int,
                 debug_display: bool = False):
        self.rom = rom
        self.movie = movie
        self.scroll_area = scroll_area
        self.start_t = start_t
        self.end_t = end_t
        self.emu = fceulib.runGame(rom)
        self.inputs1 = fceulib.readInputs(movie)
        self.inputs2 = fceulib.readInputs2(movie)
        self.debug_display = debug_display
        self.startedup = False

    def skip_steps(self, inp1s: Inputs, inp2s: Inputs):
        for i, i2 in zip(inp1s, inp2s):
            self.emu.stepFull(i, i2)

    def drive(self, inp1s: Inputs, inp2s: Inputs):
        # call read_room repeatedly until exhausting inp1s, inp2s
        while inp1s:
            room, tracks, inp1s, inp2s = self.read_room(inp1s, inp2s)
            self.refine_room_tiles(room, tracks)
            self.merge_room(room, self.rooms)
            self.last_room = rooms
        pass

    def read_room(self, inp1s: Inputs, inp2s: Inputs) -> Tuple[Room, Dict[str, Track], Inputs, Inputs):
        # Read one room at a time worth of inputs
        pass

    def dump_run(self):
        if not self.startedup:
            self.startup()
        p1inputs = self.inputs1[self.start_t:self.end_t]
        p2inputs = self.inputs2[self.start_t:end_t]
        self.ep_data = ppu_dump.ppu_output(
            self.emu,
            inputs1=p1inputs,
            inputs2=p2inputs,
            bg_data=True,
            scrolling=True,
            sprite_data=True,
            colorized_tiles=False,
            display=False,
            test_control=True,
            peekevery=4,
            scroll_area=self.scroll_area)
        return self.ep_data
# TODO no I don't like this, make it work one room at a time obviously and do sprites and everything that way too.  Time to... rewrite it in rust???  This also means ppu_dump is not quite right, since it does the run at a time.  even so, I'm ok with doing the run at a time and the mapping/rest of the pipeline room by room.

    def map_rooms(self):
        from collections import Counter
        posx, posy = 0, 0
        nt_total = {}
        potential_nt_total = {}
        nt_totals = [nt_total]
        tilex = 0
        tiley = 0
        interstitial = False
        prev = None
        curr = None
        gap = 10
        big_gap = 60
        timeSinceControl = 0
        thresh = 1
        accum = 0
        screen_offsets = {}
        room_v_time = {}
        accumX = 0
        accumY = 0
        potential_interstitial = False
        correlation_threshold = 0.1
        room_start_x = 0
        room_start_y = 0
        room_starts = {0: (0, 0)}
        previousTimeSinceControl = 0
        raw_scrolls = {}
        for t, (sx, sy) in sorted(self.ep_data["tilemap_motion"].items()):
            if t not in self.ep_data['screen_corrs']:
                self.ep_data['screen_corrs'][t] = 1
            nt = self.ep_data["nametables"][t]
            attr = self.ep_data["attr"][t]
            pal = self.ep_data['palettes'][t]
            if sx >= 16:
                sx -= 32
            if sx <= -16:
                sx += 32
            if sy >= 15:
                sy -= 30
            if sy <= -15:
                sy += 30
            tilex += sx
            tiley += sy

            raw_scrolls[t] = (tilex, tiley)
            if not self.ep_data['has_controls'][t]:
                timeSinceControl += 1
            else:
                previousTimeSinceControl = timeSinceControl
                timeSinceControl = 0
                accumX = 0
                accumY = 0
            if timeSinceControl > 0:
                accumX += sx
                accumY += sy
            if timeSinceControl > gap and ((abs(accumX) >= self.scroll_area[2] // 2
                                            or abs(accumY) >= self.scroll_area[3] // 2) or
                                           timeSinceControl > big_gap):
                if not interstitial:
                    pass
                interstitial = True
                potential_nt_total = {}
                potential_interstitial = False
            elif timeSinceControl > gap:
                potential_interstitial = True
            else:
                if True:
                    for key in potential_nt_total:
                        if key not in nt_total:
                            nt_total[key] = {}
                        for t_ in potential_nt_total[key]:
                            nt_total[key][t_] = potential_nt_total[key][t_]
                    potential_nt_total = {}
                    potential_interstitial = False
             if not (interstitial or potential_interstitial):
                 for x in range(0, self.scroll_area[2]):
                    for y in range(0, self.scroll_area[3]):
                        key = (y + tiley, x + tilex)
                        if key not in nt_total:
                            nt_total[key] = {}
                        nt_total[key][t] = (int(nt[y, x]), int(attr[y, x]), pal)
            if potential_interstitial:
                for x in range(0, self.scroll_area[2]):
                    for y in range(0, self.scroll_area[3]):
                        key = (y + tiley, x + tilex)
                        if key not in potential_nt_total:
                            potential_nt_total[key] = {}
                        potential_nt_total[key][t] = (int(nt[y, x]), int(attr[y, x]), pal)
            if (self.ep_data['screen_corrs'][t] < correlation_threshold) and self.ep_data['screen_corrs'][t] > 0:
                interstitial = True
            prev = curr
            curr = {}
            diff = 0
            for x in range(0, self.scroll_area[2]):
                for y in range(0, self.scroll_area[3]):
                    key = (y + tiley, x + tilex)
                    if key not in curr:
                        curr[key] = (int(nt[y, x]), int(attr[y, x]))
                    if prev:
                        if key not in prev:
                            diff += 1
                        elif curr[key] != prev[key]:
                            diff += 1
            if (diff > self.scroll_area[2] * self.scroll_area[3] * thresh):
                interstitial = True
            # print( t, timeSinceControl,interstitial,potential_interstitial,float(diff)/float(scroll_area[2]*scroll_area[3]),accumX,accumY)

            if ((self.ep_data['has_controls'][t] and interstitial) or (diff > self.scroll_area[2] * self.scroll_area[3] * thresh)) and (self.ep_data['screen_corrs'][t] > correlation_threshold):
                if self.debug_display:
                    plt.imshow(nt, interpolation='none')
                    plt.show()
                interstitial = False
                nt_total = {}
                nt_totals.append(nt_total)
                for t_ in range(t - previousTimeSinceControl, t):
                    room_v_time[t] = len(nt_totals) - 1
                room_starts[len(nt_totals) - 1] = (ep_data["screen_scrolls"][t][0],
                                                   ep_data["screen_scrolls"][t][1])
            room_v_time[t] = len(nt_totals) - 1
            if t in ep_data["screen_scrolls"]:
                screen_offsets[t] = (ep_data["screen_scrolls"][t][0],
                                     ep_data["screen_scrolls"][t][1])
            else:
                screen_offsets[t] = (tilex * 8, tiley * 8)
        self.screen_offsets = screen_offsets
        self.nt_totals = nt_totals
        return self.nt_totals,self.screen_offsets


def convert_image(img_buffer):
    # TODO: without allocations/reshape?
    screen = pointer_to_numpy(img_buffer)
    return screen.reshape([256, 256, 4]).astype(np.uint8)


if __name__ == "main":
    import sys
    try:
        step = sys.argv[1]
        into = sys.argv[2]
        rom = sys.argv[3]
        run = sys.argv[4]
        if len(sys.argv) > 5:
            start_t = int(sys.argv[5])
        else:
            start_t = -1
        if len(sys.argv) == 6:
            end_t = sys.argv[6]
        else:
            end_t = -1
        with open("runs.json") as runsfile:
            runs = json.load(runsfile)
            rominfo = runs[rom]
            scroll_area = rominfo["scroll_area"]
            if start_t < 0:
                runinfo = next(x for x in rominfo["runs"] if x["name"] == run)
                start_t = runinfo["start_t"]
                mappy = Mappy(rom, run, scroll_area, start_t, end_t)
    except:
        print("Usage: mappy.py step into rom run [start_t] [end_t]\n  Step: all/dump/tiles/images/sprites/tracks/sprite_anim/tile_anim")
        raise
    if step == "all" or step == "dump":
        ep_data = mappy.dump_run()
        pickle.dump(ep_data, into + "/ep_data.pkl")
    if step == "all" or step == "tiles":
        if mappy.ep_data is None:
            mappy.ep_data = pickle.load(into + "/ep_data.pkl")
        rooms = mappy.map_rooms()
        pickle.dump(rooms, into + "/room_data.pkl")
    if step == "all" or step == "images":
        if mappy.nt_totals is None or mappy.screen_offsets is None:
            mappy.nt_totals, mappy.screen_offsets = pickle.load(into + "/ep_data.pkl")
