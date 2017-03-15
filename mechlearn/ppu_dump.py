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

# 0 - Hori
# 1 - Vert
# 2 - all use 0
# 3 - all use 1
mirror_modes = {
    0: {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
    },
    1: {
        0: 0,
        1: 1,
        2: 0,
        3: 1
    },
    2: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    },
    3: {
        0: 1,
        1: 1,
        2: 1,
        3: 1
    }
 }


def nt_page(nt, nti, mirroring):
    global mirror_modes
    start = mirror_modes[mirroring][nti] * 1024
    base_nt = nt[start: start + 960]
    base_attr = nt[start + 960: start + 960 + 64]
    nt = base_nt.reshape(30, 32)
    attr = base_attr.reshape(8, 8)
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
    return nt, full_attr


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
    offset_left = 8
    offset_top = 8
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

        xScroll = emu.fc.ppu.xScroll
        yScroll = emu.fc.ppu.yScroll

        fineXScroll = xScroll & 0x7
        coarseXScroll = xScroll >> 3
        
        fineYScroll = yScroll & 0x7
        coarseYScroll = yScroll >> 3
        
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
            # TODO: maybe instead consider a span of columns on the left and middle and right and a span of rows on the top and middle and bottom, and see which of those are moving in what direction, and take the biggest/average scroll?
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
            print "Offset:", best_sx, best_sy, net_x
            motion[timestep] = (-best_sx, -best_sy)
            np_image_temp = np_image
            np_image = np_image_prev
            np_image_prev = np_image_temp

        if display:
            outputImage(emu, 'images/{}'.format(timestep), img_buffer)

        if get_bg_data:
            h_neighbors = {0: 1, 1: 0, 2: 3, 3: 2}
            v_neighbors = {0: 2, 2: 0, 1: 3, 3: 1}

            base_nti = pointer_to_numpy(emu.fc.ppu.values)[0] & 0x3
            right_nti = h_neighbors[base_nti]
            below_nti = v_neighbors[base_nti]
            right_below_nti = v_neighbors[right_nti]

            print "NTS:\n", base_nti, right_nti, "\n", below_nti, right_below_nti

            # nt
            # Getting the mirroring right and grabbing the right tile seems
            # done by the PPUTile function in fceulib's ppu.cc.
            # But it has lots of parameters including ones related to MMC5HackMode
            #  and other mapper stuff.  Because mappers are determined by mirroring!
            #  But it also relies on the global Pline to figure out which row it's in...
            #  and the `scanline` global...
            #
            nta = pointer_to_numpy(emu.fc.ppu.NTARAM)
            # change to handle other nametables?
            mirroring = emu.fc.rom.mirroring
            print mirroring, base_nti, coarseXScroll, coarseYScroll
            # 0 - Hori
            # 1 - Vert
            # 2 - all use 0
            # 3 - all use 1

            # Later, just look at VNAPages to handle roms with mappers with extra vram?
            base_nt, base_attr = nt_page(nta, base_nti, mirroring)
            right_nt, right_attr = nt_page(nta, right_nti, mirroring)
            below_nt, below_attr = nt_page(nta, below_nti, mirroring)
            right_below_nt, right_below_attr = nt_page(nta, right_below_nti, mirroring)
            
            base_rect = (coarseXScroll, coarseYScroll, 32-coarseXScroll, 30-coarseYScroll)
            right_rect = (0, coarseYScroll, coarseXScroll, 30-coarseYScroll)
            below_rect = (coarseXScroll, 0, 32-coarseXScroll, coarseYScroll)
            right_below_rect = (0, 0, coarseXScroll, coarseYScroll)
            print base_rect, right_rect
            print below_rect, right_below_rect
            print "OI"
            actualNT = np.hstack([
                np.vstack([
                    base_nt[base_rect[1]:base_rect[1]+base_rect[3],
                            base_rect[0]:base_rect[0]+base_rect[2]],
                    below_nt[below_rect[1]:below_rect[1]+below_rect[3],
                             below_rect[0]:below_rect[0]+below_rect[2]]
                ]),
                np.vstack([
                    right_nt[right_rect[1]:right_rect[1]+right_rect[3],
                             right_rect[0]:right_rect[0]+right_rect[2]],
                    right_below_nt[right_below_rect[1]:right_below_rect[1]+right_below_rect[3],
                                   right_below_rect[0]:right_below_rect[0]+right_below_rect[2]]
                ])
            ])
            actualattr = np.hstack([
                np.vstack([
                    base_attr[base_rect[1]:base_rect[1]+base_rect[3],
                              base_rect[0]:base_rect[0]+base_rect[2]],
                    below_attr[below_rect[1]:below_rect[1]+below_rect[3],
                               below_rect[0]:below_rect[0]+below_rect[2]]
                ]),
                np.vstack([
                    right_attr[right_rect[1]:right_rect[1]+right_rect[3],
                               right_rect[0]:right_rect[0]+right_rect[2]],
                    right_below_attr[right_below_rect[1]:right_below_rect[1]+right_below_rect[3],
                                     right_below_rect[0]:right_below_rect[0]+right_below_rect[2]]
                ])
            ])
            # print actualNT.shape
            # plt.imshow(actualNT)
            # plt.show()
            # print actualattr.shape
            # plt.imshow(actualattr)
            # plt.show()

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
