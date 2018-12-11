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
import random
import tensorflow as tf
import scipy.stats
from typing import cast, Iterable, Dict, Tuple, List, Optional


def dump_ppm(buf, fl):
    header = bytearray("P6\n {} {}\n 255\n".format(buf.shape[1], buf.shape[0]), "utf-8")
    ppmfile = open(fl, 'wb')
    ppmfile.write(header)

    for y in range(len(buf)):
        for x in range(len(buf[y])):
            ppmfile.write(bytearray([buf[y, x, 2], buf[y, x, 1], buf[y, x, 0]]))
    ppmfile.flush()
    ppmfile.close()


def option1_crop(w, h, fb, scroll):
    rejection_count = 32
    min_brightness_delta = 32
    for i in range(rejection_count):
        y = random.randrange(0, fb.shape[0] - h)
        x = random.randrange(0, fb.shape[1] - w)
        # TODO: a version of this where x and y are forced to be %8=0, maybe use those for my testing set??  or if performance seems bad?
        fbc = fb[y:y + h, x:x + w]
        fbgray = np.mean(fbc, -1)
        if np.max(fbgray) - np.min(fbgray) > min_brightness_delta:
            scc = (scroll[y:y + h, x:x + w] + (y, x)) % 8
            scmode = np.array((
                scipy.stats.mstats.mode(scc[:, :, 0], axis=None).mode[0],
                scipy.stats.mstats.mode(scc[:, :, 1], axis=None).mode[0]
            ))
            return fbgray / 255.0, scmode
    return None


def option1(fbs, scrolls):
    epochs = 100
    img_w = 64
    img_h = 64
    batch_size = 400
    count = fbs.shape[0]

    # set up model

    for e in range(epochs):
        # lazily create data for this batch
        patch_per_image = 8
        indices = sorted(random.sample(range(count), batch_size // patch_per_image))
        print("load")
        t = time.time()
        fb_b = fbs[indices]
        sc_b = scrolls[indices]
        print("done, dt", time.time() - t)
        in_batch = np.zeros(shape=(batch_size, img_h, img_w))
        out_batch = np.zeros(shape=(batch_size, 2))
        for i, idx in enumerate(sorted(indices * patch_per_image)):
            t = time.time()
            crop_result = option1_crop(
                img_w,
                img_h,
                fb_b[i],
                sc_b[i])
            if crop_result is not None:
                in_batch[i, :], out_batch[i, :] = crop_result
            print("dt:", time.time() - t)
        # train model with this batch
        print("batch")

    # save model params

    # test model on testing set

    # (validation set is another different game??)


with h5py.File('datagen/nes.hdf5', 'r+') as datafile:
    group = datafile['scroll/src/smb/Super Mario Bros. (JU) [!].nes/src/smb/happylee_mars608-smb-warpless.fm2/']
    scrolls = group["scrollshots"]
    fbs = group["screenshots"]
    print(scrolls.shape)
    # increase dataset size: take randomized 64x64 or 96x96 crops of all images and corresponding parts of scroll, trying up to 10 times to find a crop that isn't just a flat color.
    # increase dataset size: flip images and scrolls horizontally and vertically?
    # decrease inputs and parameters: grayscale only?
    # option 1: emit x y scroll alignment (train on mode of scrolls)
    # option 2: emit scrollshots image
    option1(fbs, scrolls)
