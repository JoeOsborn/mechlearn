import fceulib
from fceulib import VectorBytes
import numpy as np
from fceu_help import pointer_to_numpy, colorize_tile
from fceu_help import get_all_sprites, get_tile, get_sprite, outputImage
from math import log
import matplotlib.pyplot as plt
# Needed for Joe's pyenv to find CV2
import site
site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2


def convert_image(img_buffer, dest):
    # TODO: without allocations/reshape?
    screen = pointer_to_numpy(img_buffer)
    return cv2.cvtColor(screen.reshape([256, 256, 4])[:240, :, :3].astype(np.uint8),
                        cv2.COLOR_RGB2GRAY,
                        dest)


def ppu_output(emu, inputVec, **kwargs):
    start = VectorBytes()
    emu.save(start)

    peekevery = 1

    colorized2id = {}
    id2colorized = {}
    tile2colorized = {}
    data = []
    nametables = {}
    nametables2 = {}
    nametable_outputs = []
    attr_outputs = []
    xScrolls = None
    motion = {}

    img_buffer = VectorBytes()
    np_image = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_prev = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_temp = None
    get_bg_data = kwargs.get("bg_data", True)
    get_colorized_tiles = kwargs.get("colorized_tiles", True)
    get_sprite_data = kwargs.get("sprite_data", True)
    get_scroll = kwargs.get("scrolling", True)

    if get_scroll:
        emu.imageInto(img_buffer)
        convert_image(img_buffer, np_image_prev)

    display = kwargs.get("display", True)
    net_x = 0
    # assume scrolling < K px per frame
    scroll_window = 5
    offset_left = 4
    offset_top = 4
    motion = {}
    for timestep, inp in enumerate(inputVec):
        emu.stepFull(inp, 0x0)
        if get_scroll:
            emu.imageInto(img_buffer)
            # TODO: without allocations?
            convert_image(img_buffer, np_image)
        if not (timestep % peekevery == 0):
            continue
        if xScrolls is not None:
            xScrolls[timestep] = emu.xScroll

        # nametable scrolling stuff, worry about it later
        xScroll = emu.fc.ppu.xScroll
        yScroll = emu.fc.ppu.yScroll
        fineXScroll = xScroll & 0x7
        coarseXScroll = xScroll >> 3

        # What is scrolling?
        # There's two parts:

        # * visually, what is moving around on the screen?
        # * Which parts of which nametables are visible?

        # The X and Y registers are not super authoritative because if
        # the screen is split into sections of if special effects are
        # present, they might change arbitrarily during rendering and
        # their values at frame end may be arbitrary.  So we have to
        # do something visual.

        # The challenge for us is to figure out, first of all, what "real" or
        # "perceptual" scrolling is happening, and then later to figure out
        # what parts of what nametables are visible due to that.  So we start
        # by figuring out screen motion by looking at the emulator's
        # framebuffer.

        if get_scroll and timestep > 0:
            result = cv2.matchTemplate(
                np_image,
                np_image_prev[offset_top:240 - offset_top * 2,
                              offset_left:256 - offset_left * 2],
                cv2.TM_CCOEFF_NORMED
            )
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(result)
            # print minv, maxv, minloc, maxloc
            best_sx, best_sy = 0, 0
            cx, cy = offset_left, offset_top
            best_match = result[cy, cx]
            # Look around the center of the image.  does it get better-matched
            # going to the left, right, up, or down?
            for sx in range(-scroll_window, scroll_window):
                for sy in range(-scroll_window, scroll_window):
                    match = result[cy + sy, cx + sx]
                    if match > best_match:
                        best_sx = sx
                        best_sy = sy
                        best_match = match

            net_x -= best_sx
            # print "Offset:", best_sx, best_sy, net_x
            motion[timestep] = (-best_sx, -best_sy)
            np_image_temp = np_image
            np_image = np_image_prev
            np_image_prev = np_image_temp

        if display:
            outputImage(emu, 'images/{}'.format(timestep), img_buffer)

        nt_index = pointer_to_numpy(emu.fc.ppu.values)[0] & 0x3
        if get_bg_data:
            # nt
            nt = pointer_to_numpy(emu.fc.ppu.NTARAM)
            attr = nt[(nt_index * 1024 + 960):(nt_index * 1024 + 1024)]
            nt = nt[(nt_index * 1024):(nt_index * 1024 + 960)]
            nt = nt.reshape(30, 32)
            attr = attr.reshape(8, 8)
            full_attr = np.zeros([16, 16])
            for xx in range(8):
                for yy in range(8):
                    val = attr[yy, xx]
                    tr = (val & 0x0c) >> 2
                    tl = val & 0x03
                    br = (val & 0xc0) >> 6
                    bl = (val & 0x30) >> 4
                    full_attr[yy * 2, xx * 2] = tl
                    full_attr[yy * 2, xx * 2 + 1] = tr
                    full_attr[yy * 2 + 1, xx * 2] = bl
                    full_attr[yy * 2 + 1, xx * 2 + 1] = br
            full_attr = full_attr[:15, :]
            full_attr = np.kron(full_attr, np.ones((2, 2)))
            nametables = (nt, full_attr)
            nt_index = 1 - nt_index
            # nt2
            nt = pointer_to_numpy(emu.fc.ppu.NTARAM)
            attr = nt[(nt_index * 1024 + 960):(nt_index * 1024 + 1024)]
            nt = nt[(nt_index * 1024):(nt_index * 1024 + 960)]
            nt = nt.reshape(30, 32)
            attr = attr.reshape(8, 8)
            full_attr = np.zeros([16, 16])
            for xx in range(8):
                for yy in range(8):
                    val = attr[yy, xx]
                    tr = (val & 0x0c) >> 2
                    tl = val & 0x03
                    br = (val & 0xc0) >> 6
                    bl = (val & 0x30) >> 4
                    full_attr[yy * 2, xx * 2] = tl
                    full_attr[yy * 2, xx * 2 + 1] = tr
                    full_attr[yy * 2 + 1, xx * 2] = bl
                    full_attr[yy * 2 + 1, xx * 2 + 1] = br
            full_attr = full_attr[:15, :]
            full_attr = np.kron(full_attr, np.ones((2, 2)))
            nametables2 = (nt, full_attr)

            # TODO: use yScroll also?  Handle other mirroring/wrapping modes?
            NT = np.kron(nametables[0], np.ones((8, 8)))
            NT2 = np.kron(nametables2[0], np.ones((8, 8)))
            actualNT = np.zeros(NT.shape)
            actualNT[:, :(NT.shape[1] - xScroll)] = NT[:, xScroll:]
            actualNT[:, (NT.shape[1] - xScroll):] = NT2[:, :xScroll]

            attr = np.kron(nametables[1], np.ones((8, 8)))
            attr2 = np.kron(nametables2[1], np.ones((8, 8)))

            actualattr = np.zeros(NT.shape)
            actualattr[:, :(attr.shape[1] - xScroll)] = attr[:, xScroll:]
            actualattr[:, (attr.shape[1] - xScroll):] = attr2[:, :xScroll]
            pairs = set()
            if get_colorized_tiles:
                pt = pointer_to_numpy(emu.fc.ppu.PALRAM)
                for ii in range(actualattr.shape[0]):
                    for jj in range(actualattr.shape[1]):
                        pairs.add((int(actualNT[ii, jj]),
                                   int(actualattr[ii, jj])))
                for pair in pairs:
                    if pair not in tile2colorized:
                        tile2colorized[pair] = colorize_tile(
                            get_tile(pair[0], emu.fc),
                            pair[1],
                            pt)[:, :, :3]
                        # Have to divide by 255 to actually
                        # display with plt.imshow
            nametable_outputs.append(actualNT)
            attr_outputs.append(actualattr)

        if get_sprite_data:
            sprite_list, colorized_sprites = get_all_sprites(emu.fc)
            for sprite_id, sprite in enumerate(sprite_list):

                if np.sum(colorized_sprites[sprite_id].ravel()) == 0:
                    continue
                uniq = tuple(colorized_sprites[sprite_id].ravel())
                if uniq not in colorized2id:
                    colorized2id[uniq] = len(colorized2id)
                    # plt.imshow(colorized_sprites[sprite_id][:,:,:3]/255.)
                    # plt.show()
                    id2colorized[
                        colorized2id[uniq]
                    ] = colorized_sprites[sprite_id]
                # print timestep,  colorized2id[uniq], sprite[:2]
                data.append((timestep, colorized2id[uniq], sprite))

    emu.load(start)
    results = {}
    if get_scroll:
        results["screen_motion"] = motion
    if get_bg_data:
        results["nametables"] = nametable_outputs
        results["attrs"] = attr_outputs
        if get_colorized_tiles:
            results["tile2colorized"] = tile2colorized
    if get_sprite_data:
        results["id2colorized"] = id2colorized
        results["colorized2id"] = colorized2id
        results["sprite_data"] = data
    return results
