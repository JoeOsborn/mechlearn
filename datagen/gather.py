# type check: MYPYPATH=../mypy-data/numpy-mypy mypy tester.py
# run: python tester.py
import h5py
import matplotlib.pyplot as plt
import subprocess
import atexit
import struct
import io
import time
import numpy as np
from typing import cast, Iterable, Dict, Tuple, List, Optional
from remocon import Mesen, Infos
import remocon


def dump_ppm(buf, fl):
    header = bytearray("P6\n {} {}\n 255\n".format(buf.shape[1], buf.shape[0]), "utf-8")
    ppmfile = open(fl, 'wb')
    ppmfile.write(header)

    for y in range(len(buf)):
        for x in range(len(buf[y])):
            ppmfile.write(bytearray([buf[y, x, 2], buf[y, x, 1], buf[y, x, 0]]))
    ppmfile.flush()
    ppmfile.close()


def process_play(romfile, fm2file):
    controls_fm = remocon.read_fm2("datagen/" + fm2file)
    controls = list(map(remocon.moves_to_bytes, controls_fm))
    ctrl_count = len(controls[0])
    romkey = romfile.replace("/", "_").replace(" ", "-")
    fm2key = fm2file.replace("/", "_").replace(" ", "-")
    with h5py.File('datagen/nes.hdf5', 'r+') as datafile:
        screenshotsds = datafile.create_dataset('scroll/{}/{}/screenshots'.format(romkey, fm2key), (ctrl_count, 240, 256, 3), dtype='uint8', compression="gzip", compression_opts=9, shuffle=True)
        scrollshotsds = datafile.create_dataset('scroll/{}/{}/scrollshots'.format(romkey, fm2key), (ctrl_count, 240, 256, 2), dtype='float', compression="gzip", compression_opts=9, shuffle=True)

        print(ctrl_count)

        remo = Mesen("mesen/remocon/obj.x64/remocon", "datagen/" + romfile)
        last_frame_data = None
        span = 200
        screenshots = np.zeros((span, 240, 256, 3), dtype='uint8')
        scrollshots = np.zeros((span, 240, 256, 2), dtype='uint8')

        for steps in range(0, ctrl_count, span):
            step_end = min(ctrl_count, steps + span)
            start_t = time.time()
            results = remo.step(
                [controls[0][steps:step_end]],
                Infos(framebuffer=True,
                      live_sprites=True,
                      tiles_by_pixel=True,
                      new_tiles=True, new_sprite_tiles=True))
            print("DT:", time.time() - start_t, "Progress:", step_end, "/", ctrl_count)
            start_t = time.time()
            for i, pf in enumerate(results[0]):
                if i % 100 == 0:
                    print("Step", steps + i)
                screenshots[i] = pf.framebuffer
                h = len(pf.tiles_by_pixel)
                w = len(pf.tiles_by_pixel[0])
                for y in range(h):
                    for x in range(w):
                        pt = pf.tiles_by_pixel[y][x]
                        scrollshots[i, y, x, 0] = pt.y_scroll
                        scrollshots[i, y, x, 1] = pt.x_scroll
            print("Dataset transformed, dt:", time.time() - start_t)
            start_t = time.time()
            last_frame_data = pf
            screenshotsds[steps:step_end] = screenshots[:step_end - steps]
            scrollshotsds[steps:step_end] = scrollshots[:step_end - steps]
            print("Data written", time.time() - start_t)


# process_play(
    # "src/smb/Super Mario Bros. (JU) [!].nes",
    # "src/smb/happylee_mars608-smb-warpless.fm2"
# )
process_play(
    "src/loz/Legend of Zelda, The (U) (PRG0) [!].nes",
    "src/loz/taseditorv1-legendofzelda-allitems.fm2"
)
