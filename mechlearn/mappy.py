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
from tracking import Track, Tracks
from ppu_dump import RowCol, PPUDump, DXY, ColoredPattern, Time


# Type aliases
Rect = Tuple[int, int, int, int]
Inputs = Sequence[int]
Tile = Tuple[int, int, int]
Room = Dict[RowCol, Dict[Time, Tile]]
RoomIdx = int


class Mappy:
    def __init__(self,
                 rom: str, movie: str,
                 scroll_area: Rect, start_t: int, end_t: int,
                 debug_display: bool = False,
                 into_dir: str = "./"):
        self.rom = rom
        self.movie = movie
        self.scroll_area = scroll_area
        self.start_t = start_t
        self.end_t = end_t
        self.emu = fceulib.runGame(rom)
        self.inputs1 = fceulib.readInputs(movie)
        self.inputs2 = fceulib.readInputs2(movie)
        self.debug_display = debug_display
        self.ep_data: PPUDump = {}
        self.into_dir = into_dir
        self.rooms: List[Room] = []
        self.room_v_time: Dict[Time, RoomIdx] = {}
        self.room_starts: Dict[RoomIdx, DXY] = {}

    def dump_run(self) -> PPUDump:
        for i, i2 in zip(self.inputs1[:self.start_t], self.inputs2[:self.start_t]):
            self.emu.stepFull(i, i2)
        self.ep_data = ppu_dump.ppu_output(
            self.emu,
            self.inputs1[self.start_t:self.end_t],
            inputs2=self.inputs2[self.start_t:self.end_t],
            bg_data=True,
            scrolling=True,
            sprite_data=True,
            colorized_tiles=False,
            display=False,
            test_control=True,
            peekevery=1,
            scroll_area=self.scroll_area,
            debug_output=self.debug_display)
        return self.ep_data

    def dump_room(self, nt_total: Room, key: str):
        all_spots = list(nt_total.keys())
        minx = min(map(lambda pt: pt[1], all_spots))
        maxx = max(map(lambda pt: pt[1], all_spots)) + 1
        miny = min(map(lambda pt: pt[0], all_spots))
        maxy = max(map(lambda pt: pt[0], all_spots)) + 1
        nt_result = {}
        for x in range(minx, maxx):
            for y in range(miny, maxy):
                moments = nt_total[(y, x)]
                nt_result[y - miny, x - minx] = moments

                colorized = self.ep_data["tile2colorized"]
        out_image = np.zeros(shape=((maxy - miny) * 8, (maxx - minx) * 8, 4))
        for x in range(0, maxx - minx):
            for y in range(0, maxy - miny):
                col = np.zeros(shape=(8, 8, 3))
                if (y, x) in nt_result:
                    ind_per_tile = nt_result[y, x].values()
                    i2c: Dict[Tile, int] = {}
                    for i in ind_per_tile:
                        if i not in i2c:
                            i2c[i] = 0
                        i2c[i] += 1
                    col = colorized[sorted(i2c.items(), key=lambda entry: (entry[1], entry[0]))[-1][0]]
                out_image[y * 8:(y + 1) * 8, x * 8:(x + 1) * 8, :3] = col / 255.
                out_image[y * 8:(y + 1) * 8, x * 8:(x + 1) * 8, 3] = 1.0
        import os
        image_path = os.path.join(self.into_dir, key + ".png")
        os.makedirs(os.path.dirname(image_path) + "/", exist_ok=True)
        plt.figure(figsize=(20, 10))
        plt.imshow(out_image)
        plt.savefig(image_path)
        # outImg = Image.fromBytes("RGBA",
        #                          (maxx-minx, maxy-miny),
        #                          str(bytearray(out_image)))

    # def drive(self, inp1s: Inputs, inp2s: Inputs):
    #     # call read_room repeatedly until exhausting inp1s, inp2s
    #     while inp1s:
    #         room, tracks, inp1s, inp2s = self.read_room(inp1s, inp2s)
    #         self.refine_room_tiles(room, tracks)
    #         self.merge_room(room)
    #     pass

    # def read_room(self, inp1s: Inputs, inp2s: Inputs) -> Tuple[Room, Tracks, Inputs, Inputs]:
    #     return ({}, {}, [], [])

    # def refine_room_tiles(self, room, tracks):
    #     pass

    def merge_room(self, room: Room, tracks: Tracks, tidx_start: int, tidx_end: int, times: List[Time]):
        self.rooms.append(room)
        for t in range(tidx_start, tidx_end):
            self.room_v_time[t] = len(self.rooms) - 1
        t0 = times[tidx_start]
        print("Merge room:", len(self.rooms) - 1, tidx_start, t0, tidx_end - 1, times[tidx_end - 1])
        sx = 0 if t0 == 0 else self.ep_data["screen_scrolls"][t0][0]
        sy = 0 if t0 == 0 else self.ep_data["screen_scrolls"][t0][1]
        self.room_starts[len(self.rooms) - 1] = (sx, sy)

    def map_rooms(self) -> List[Room]:
        # map up to some t
        times = sorted(self.ep_data["tilemap_motion"].keys())
        tidx = 0
        print("Noticed steps:", len(times))
        while tidx < len(times):
            room, tracks, tidx1 = self.read_room(tidx, times)
            self.merge_room(room, tracks, tidx, tidx1, times)
            self.dump_room(self.rooms[-1], "room_premerge/" + str(len(self.rooms) - 1))
            tidx = tidx1
        return self.rooms

    def read_room(self, tstartidx: int, times: Sequence[Time]) -> Tuple[Room, Tracks, int]:
        tidx = tstartidx
        nt_total: Room = {}
        potential_nt_total: Room = {}
        tilex = 0
        tiley = 0
        interstitial = False
        prev: Dict[RowCol, Tuple[int, int]] = {}
        curr: Dict[RowCol, Tuple[int, int]] = {}
        gap = 10
        big_gap = 90
        timeSinceControl = 0
        thresh = 1
        accumX = 0
        accumY = 0
        potential_interstitial = False
        correlation_threshold = 0.1
        print("Entering readroom loop")
        while tidx < len(times):
            if tidx > 0 and tidx % 1000 == 0:
                self.dump_room(nt_total, "room_dbg/" + str(tidx))
            t = times[tidx]
            print("t:", t)
            (sx, sy) = self.ep_data["tilemap_motion"][t]
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

            if not self.ep_data['has_controls'][t]:
                timeSinceControl += 1
                print("NC")
            else:
                timeSinceControl = 0
                accumX = 0
                accumY = 0
            if timeSinceControl > 0:
                print("sxsy", sx, sy)
                accumX += sx
                accumY += sy
            if (timeSinceControl > gap and
                (abs(accumX) >= self.scroll_area[2] // 2
                 or abs(accumY) >= self.scroll_area[3] // 2)) or timeSinceControl > big_gap:
                print("internoticed", timeSinceControl, big_gap, accumX, accumY)
                interstitial = True
                potential_nt_total = {}
                potential_interstitial = False
            elif timeSinceControl > gap:
                potential_interstitial = True
            else:
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
            interesting_screen = (self.ep_data['screen_corrs'][t] > correlation_threshold)
            prev = curr
            curr = {}
            diff = 0
            for x in range(0, self.scroll_area[2]):
                for y in range(0, self.scroll_area[3]):
                    key = (y + tiley, x + tilex)
                    if key not in curr:
                        curr[key] = (int(nt[y, x]), int(attr[y, x]))
                    if key not in prev:
                        diff += 1
                    elif curr[key] != prev[key]:
                        diff += 1
            if (diff > self.scroll_area[2] * self.scroll_area[3] * thresh):
                interstitial = True
            # print( t, timeSinceControl,interstitial,potential_interstitial,float(diff)/float(scroll_area[2]*scroll_area[3]),accumX,accumY)
            interstitial_over = (self.ep_data['has_controls'][t] and interstitial)
            sudden_change = (diff > self.scroll_area[2] * self.scroll_area[3] * thresh)
            end_of_room = (interstitial_over or sudden_change) and interesting_screen
            if end_of_room:
                print("Room ended:\nInterstitial:", interstitial, "Over:", interstitial_over, "\nSudden change:", sudden_change, "\nInteresting:", interesting_screen)
                return nt_total, {}, tidx + 1
            tidx += 1
        print("Room ended at end of observations")
        return nt_total, {}, tidx


def convert_image(img_buffer: Sequence[int]) -> np.ndarray:
    # TODO: without allocations/reshape?
    screen = pointer_to_numpy(img_buffer)
    return screen.reshape([256, 256, 4]).astype(np.uint8)


if __name__ == "__main__":
    def do_main():
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
            if len(sys.argv) > 6:
                end_t = int(sys.argv[6])
            else:
                end_t = -1
            if len(sys.argv) > 7:
                debug_display = sys.argv[6].lower() == "true" or sys.argv[6] == "1"
            else:
                debug_display = False
            with open("runs.json") as runsfile:
                runs = json.load(runsfile)
                rominfo = runs[rom]
                scroll_area = rominfo["scroll_area"]
                if start_t < 0:
                    runinfo = next(x for x in rominfo["runs"] if x["name"] == run)
                    start_t = runinfo["start_t"]
            print("Config", rom, run, scroll_area, start_t, end_t, into)
            mappy = Mappy(rom, run, scroll_area, start_t, end_t, debug_display, into)
        except:
            print("Usage: mappy.py step into rom run [start_t] [end_t]\n  Step: all/dump/map")
            raise
        if step == "all" or step == "dump":
            print("Dumping PPU data...")
            ep_data = mappy.dump_run()
            with open(into + "/ep_data.pkl", 'wb') as f:
                pickle.dump(ep_data, f)
        if step == "all" or step == "map":
            print("Mapping out rooms...")
            if not mappy.ep_data:
                with open(into + "/ep_data.pkl", 'rb') as f:
                    mappy.ep_data = pickle.load(f)
            rooms = mappy.map_rooms()
            with open(into + "/room_data.pkl", 'wb') as f:
                pickle.dump(rooms, f)
    do_main()
