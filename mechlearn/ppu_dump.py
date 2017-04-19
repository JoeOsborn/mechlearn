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


def convert_image(img_buffer, col=cv2.COLOR_RGB2GRAY):
    # TODO: without allocations/reshape?
    screen = pointer_to_numpy(img_buffer)
    return screen.reshape([256, 256, 4])[:240, :, :3].astype(np.uint8)

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
    nametable_outputs = []
    attr_outputs = []
    scrolled_nt_outputs = []
    scrolled_attr_outputs = []
    xScrolls = None
    motion = {}
    scrolls = {}
    tm_scrolls = {}
    tm_motion = {}

    img_buffer = VectorBytes()
    np_image = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_prev = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_temp = None
    get_bg_data = kwargs.get("bg_data", True)
    scroll_area = kwargs.get("scroll_area", (0, 0, 32, 30))
    get_sprite_data = kwargs.get("sprite_data", True)
    get_scroll = kwargs.get("scrolling", True) or get_bg_data
    test_control = kwargs.get("test_control", False) 
    

    if get_scroll:
        emu.imageInto(img_buffer)
        np_image_prev = convert_image(img_buffer)
        big_picture = np.zeros(shape=(240*2, 256*2, 3))

    if test_control:
        
        start_state = fceulib.VectorBytes()
    display = kwargs.get("display", True)
    net_x = 0
    net_y = 0
    offset_left = 8
    offset_top = 8


    has_controls = {}
    for timestep, inp in enumerate(inputVec):

        if not (timestep % peekevery == 0):
            continue
        
        if test_control:
            images = []
            emu.save(start_state)
            emu.stepFull(inp, 0x0)
            next = timestep + 1
            if next >= len(inputVec):
                next = len(inputVec)-1
            emu.stepFull(inputVec[next],0x0)            
            emu.imageInto(img_buffer)
            np_image = convert_image(img_buffer)           
            
            has_control = False
            for test_inp in [0,1,4,5,6,7]:
                emu.load(start_state)                
                emu.stepFull(1 << test_inp,0x0)
                emu.stepFull(inputVec[next],0x0)   
                emu.imageInto(img_buffer)
                image = convert_image(img_buffer)
                
                if np.sum(np.abs(image - np_image)) > 0:
                    has_control = True
                    break
            
            has_controls[timestep] = has_control
            emu.load(start_state)            
        
        emu.stepFull(inp, 0x0)
        
        if get_scroll or get_bg_data or test_control:
            emu.imageInto(img_buffer)
            # TODO: without allocations?
            np_image = convert_image(img_buffer)

        if get_scroll and timestep > 0:
            # TODO: maybe instead consider a span of columns on the left and middle and right and a span of rows on the top and middle and bottom, and see which of those are moving in what direction, and take the biggest/average scroll?
            result = cv2.matchTemplate(
                np_image_prev[scroll_area[1]*8:(scroll_area[1]+scroll_area[3])*8,
                              scroll_area[0]*8:(scroll_area[0]+scroll_area[2])*8],
                np_image[scroll_area[1]*8+offset_top:scroll_area[1]*8+scroll_area[3]*8-offset_top*2,
                         scroll_area[0]*8+offset_left:scroll_area[0]*8+scroll_area[2]*8-offset_left*2],
                cv2.TM_CCOEFF_NORMED
            )
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(result)
            print "Match1", minv, maxv, minloc, maxloc
            best_sx, best_sy = 0, 0
            cx, cy = offset_left, offset_top
            scroll_window = 5
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
            net_x += best_sx
            net_y += best_sy
            print "Sc1 Offset:", best_sx, best_sy, net_x, net_y
            motion[timestep] = (best_sx, best_sy)
            scrolls[timestep] = (net_x, net_y)

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
            mirroring = emu.fc.cart.mirroring
            print "M", mirroring, "base", base_nti
            # 0 - Hori
            # 1 - Vert
            # 2 - all use 0
            # 3 - all use 1

            # Later, just look at VNAPages to handle roms with mappers with extra vram?
            base_nt, base_attr = nt_page(nta, base_nti, mirroring)
            right_nt, right_attr = nt_page(nta, right_nti, mirroring)
            below_nt, below_attr = nt_page(nta, below_nti, mirroring)
            right_below_nt, right_below_attr = nt_page(nta, right_below_nti, mirroring)
            
            fullNTs = np.vstack([
                np.hstack([
                    base_nt,
                    right_nt
                ]),
                np.hstack([
                    below_nt,
                    right_below_nt
                ])
            ])
            fullAttr = np.vstack([
                np.hstack([
                    base_attr,
                    right_attr
                ]),
                np.hstack([
                    below_attr,
                    right_below_attr
                ])
            ])
                        
            nametable_outputs.append(fullNTs)
            attr_outputs.append(fullAttr)

            pairs = set()
            pt = pointer_to_numpy(emu.fc.ppu.PALRAM)
            for ii in range(fullAttr.shape[0]):
                for jj in range(fullAttr.shape[1]):
                    pairs.add((int(fullNTs[ii, jj]),
                               int(fullAttr[ii, jj])))
            for pair in pairs:
                if pair not in tile2colorized:
                    tile2colorized[pair] = colorize_tile(
                        get_tile(pair[0], emu.fc),
                        pair[1],
                        pt)[:, :, :3]
                    # Have to divide by 255 to actually
                    # display with plt.imshow

            # OK, let's figure out our scrolling situation.
            # First build a mighty template image out of the whole picture.
            # We're gonna template match to see how the baby real image fits inside
            # the big full screen image.
            # We can't really use the scroll info determined earlier, because we don't know
            # the x and y scroll as of initialization time
            for ii in range(fullAttr.shape[0]):
                for jj in range(fullAttr.shape[1]):
                    pair = (int(fullNTs[ii, jj]),
                            int(fullAttr[ii, jj]))
                    big_picture[ii*8:ii*8+8, jj*8:jj*8+8, :] = tile2colorized[pair]/255.0

            if timestep % 60 == 0:
                print "T:",timestep
                plt.imshow(big_picture)
                plt.show()
                plt.imshow(np_image[scroll_area[1]*8:(scroll_area[1]+scroll_area[3])*8,
                                    scroll_area[0]*8:(scroll_area[0]+scroll_area[2])*8])
                plt.show()

            insets = cv2.matchTemplate(
                big_picture.astype(np.uint8),
                np_image[scroll_area[1]*8:(scroll_area[1]+scroll_area[3])*8,
                         scroll_area[0]*8:(scroll_area[0]+scroll_area[2])*8],
                cv2.TM_CCOEFF_NORMED
            )
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(insets)
            sx = maxloc[0]/8
            sy = maxloc[1]/8
            print "Sc2:", sx, sy
            #plt.imshow(np.tile(fullNTs, (2, 2))[sy:sy+scroll_area[3], sx:sx+scroll_area[2]])
            #plt.show()
            # TODO: test this!
            scrolled_nt_outputs.append(fullNTs[sy:sy+scroll_area[3],
                                               sx:sx+scroll_area[2]])
            scrolled_attr_outputs.append(fullAttr[sy:sy+scroll_area[3],
                                                  sx:sx+scroll_area[2]])
            tm_scrolls[timestep] = (sx, sy)
            px, py = tm_scrolls[timestep-1] if timestep > 0 else (sx, sy)
            mx, my = (sx-px, sy-py)
            if mx >= 16:
                mx -= 32
            if mx <= -16:
                mx += 32
            if my >= 15:
                my -= 30
            if my <= -15:
                my += 30
            tm_motion[timestep] = (mx, my)
            print sx, sy, px, py, tm_motion[timestep]
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
        np_image_temp = np_image
        np_image = np_image_prev
        np_image_prev = np_image_temp

    emu.load(start)
    results = {}
    if get_scroll:
        results["screen_motion"] = motion
        results["screen_scrolls"] = scrolls
    if get_bg_data:
        results["full_nametables"] = nametable_outputs
        results["full_attrs"] = attr_outputs
        results["nametables"] = scrolled_nt_outputs
        results["attr"] = scrolled_attr_outputs
        results["tile2colorized"] = tile2colorized
        results["tilemap_motion"] = tm_motion
        results["tilemap_scrolls"] = tm_scrolls
    if get_sprite_data:
        results["id2colorized"] = id2colorized
        results["colorized2id"] = colorized2id
        results["sprite_data"] = data
    if test_control:
        results['has_controls'] = has_controls
    return results
