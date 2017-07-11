import fceulib
from fceulib import VectorBytes
import numpy as np
import random
from fceu_help import pointer_to_numpy, colorize_tile
from fceu_help import get_all_sprites, get_tile, get_sprite, outputImage
from math import log
if __name__ != "__main__":
    import matplotlib.pyplot as plt
# Needed for Joe's pyenv to find CV2
import site
site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2


def convert_image(img_buffer, col=cv2.COLOR_RGB2GRAY):
    # TODO: without allocations/reshape?
    
    screen = pointer_to_numpy(img_buffer, copy=True)
    

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


def test_control_(emu, start_state,
                  inp, inp2, timestep, inputVec,
                  img_buffer, img_buffer2):
    emu.save(start_state)
    
    emu.stepFull(inp, inp2)
    
    steps = 3
    next = timestep
    for ii in range(steps):
        next = next + 1
        if next >= len(inputVec):
            next = len(inputVec) - 1
        emu.stepFull(inputVec[next], inp2)
    
    emu.imageInto(img_buffer)
    
    has_control = False
    inps = [0, 1, 4, 5, 6, 7]
    # Might save some simulation steps?
    random.shuffle(inps)
    for test_inp in inps:
        emu.load(start_state)
        emu.stepFull(1 << test_inp, 0x0)
        for ii in range(steps):
            emu.stepFull(1 << test_inp, 0x0)
        emu.imageInto(img_buffer2)
        np1 = np.array(img_buffer, copy=False)
        np2 = np.array(img_buffer2, copy=False)
        has_control = np.sum(np.abs(np1 - np2)) > 0
        if has_control:
            break
    emu.load(start_state)
    
    return has_control


def test_scrolling_visual_(np_image_prev, np_image, net_x, net_y,
                           offset_left, offset_top, scroll_area):
    # TODO: maybe instead consider a span of columns on the left and
    # middle and right and a span of rows on the top and middle and
    # bottom, and see which of those are moving in what direction, and
    # take the biggest/average scroll?
    result = cv2.matchTemplate(
        np_image_prev[scroll_area[1] * 8:(scroll_area[1] + scroll_area[3]) * 8,
                      scroll_area[0] * 8:(scroll_area[0] + scroll_area[2]) * 8],
        np_image[scroll_area[1] * 8 + offset_top:scroll_area[1] * 8 + scroll_area[3] * 8 - offset_top * 2,
                 scroll_area[0] * 8 + offset_left:scroll_area[0] * 8 + scroll_area[2] * 8 - offset_left * 2],
        cv2.TM_CCOEFF_NORMED
    )

    minv, maxv, minloc, maxloc = cv2.minMaxLoc(result)
    # print "Match1", minv, maxv, minloc, maxloc
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
    # print "Sc1 Offset:", best_sx, best_sy, net_x, net_y
    return (best_sx, best_sy), (net_x, net_y), maxv-minv


def test_bg_data_full_(emu, tile2colorized):
    h_neighbors = {0: 1, 1: 0, 2: 3, 3: 2}
    v_neighbors = {0: 2, 2: 0, 1: 3, 3: 1}

    base_nti = emu.fc.ppu.values[0] & 0x3
    right_nti = h_neighbors[base_nti]
    below_nti = v_neighbors[base_nti]
    right_below_nti = v_neighbors[right_nti]

    # print "NTS:\n", base_nti, right_nti, "\n", below_nti, right_below_nti

    # nt
    # Getting the mirroring right and grabbing the right tile seems
    # done by the PPUTile function in fceulib's ppu.cc.
    # But it has lots of parameters including ones related to MMC5HackMode
    #  and other mapper stuff.  Because mappers are determined by mirroring!
    #  But it also relies on the global Pline to figure out which row it's in...
    #  and the `scanline` global...
    #
    nta = pointer_to_numpy(emu.fc.ppu.NTARAM, copy=False)
    # change to handle other nametables?
    mirroring = emu.fc.cart.mirroring
    # print "M", mirroring, "base", base_nti
    # 0 - Hori
    # 1 - Vert
    # 2 - all use 0
    # 3 - all use 1

    # Later, just look at VNAPages to handle roms with mappers with
    # extra vram?
    base_nt, base_attr = nt_page(nta, base_nti, mirroring)
    right_nt, right_attr = nt_page(nta, right_nti, mirroring)
    below_nt, below_attr = nt_page(nta, below_nti, mirroring)
    right_below_nt, right_below_attr = nt_page(
        nta, right_below_nti, mirroring)

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

    pairs = set()
    pt = pointer_to_numpy(emu.fc.ppu.PALRAM, copy=False)
    pt_id = tuple(pt.ravel())
    for ii in range(fullAttr.shape[0]):
        for jj in range(fullAttr.shape[1]):
            pairs.add((int(fullNTs[ii, jj]),
                       int(fullAttr[ii, jj]), pt_id))
    for pair in pairs:
        if pair not in tile2colorized:
            tile2colorized[pair] = colorize_tile(
                get_tile(pair[0], emu.fc),
                pair[1],
                pt)[:, :, :3]
            # Have to divide by 255 to actually
            # display with plt.imshow
    return fullNTs, fullAttr, pt_id


def fill_big_picture(nts, attrs, pal, tile2colorized, big_picture):
    for ii in range(attrs.shape[0]):
        for jj in range(attrs.shape[1]):
            pair = (int(nts[ii, jj]),
                    int(attrs[ii, jj]),
                    pal)
            big_picture[ii * 8:ii * 8 + 8,
                        jj * 8:jj * 8 + 8,
                        :] = tile2colorized[pair]
    big_picture /= 255.0


def test_bg_data_scrolled_(nts, attrs, pal, tile2colorized,
                           scroll_area, timestep, last_tm_scroll,
                           big_picture, np_image, debug_output):
    # OK, let's figure out our scrolling situation.
    # First build a mighty template image out of the whole picture.
    # We're gonna template match to see how the baby real image fits inside
    # the big full screen image.
    # We can't really use the scroll info determined earlier, because we don't know
    # the x and y scroll as of initialization time
    fill_big_picture(nts, attrs, pal, tile2colorized, big_picture)
    center = True
    if center:
        insets = cv2.matchTemplate(
            cv2.cvtColor(
                (big_picture * 128 + 127).astype(np.uint8),
                cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(
                (np_image[scroll_area[1] * 8:
                          (scroll_area[1] + scroll_area[3]) * 8,
                          scroll_area[0] * 8:
                          (scroll_area[0] + scroll_area[2]) * 8] / 2 + 128
                 ).astype(np.uint8),
                cv2.COLOR_BGR2GRAY),
            cv2.TM_CCOEFF_NORMED
        )
    else:
        insets = cv2.matchTemplate(
            cv2.cvtColor(
                (big_picture * 255).astype(np.uint8),
                cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(
                (np_image[scroll_area[1] * 8:
                          (scroll_area[1] + scroll_area[3]) * 8,
                          scroll_area[0] * 8:
                          (scroll_area[0] + scroll_area[2]) * 8]
                 ).astype(np.uint8),
                cv2.COLOR_BGR2GRAY),
            cv2.TM_CCOEFF_NORMED
        )
    minv, maxv, minloc, maxloc = cv2.minMaxLoc(insets)
    sx = maxloc[0] / 8
    sy = maxloc[1] / 8
    if timestep % 60 == 0 and debug_output:
        print "T:", timestep
        plt.imshow(insets)
        plt.show()
        plt.imshow(cv2.cvtColor(
            (big_picture * 128 + 127).astype(np.uint8), cv2.COLOR_BGR2GRAY))
        plt.plot((sx * 8, (sx + scroll_area[2]) * 8,
                  (sx + scroll_area[2]) * 8, sx * 8, sx * 8),
                 (sy * 8, sy * 8, (sy + scroll_area[3]) * 8,
                  (sy + scroll_area[3]) * 8, sy * 8), 'r')
        plt.show()
        plt.imshow(
            cv2.cvtColor(
                (np_image[scroll_area[1] * 8:
                          (scroll_area[1] + scroll_area[3]) * 8,
                          scroll_area[0] * 8:
                          (scroll_area[0] + scroll_area[2]) * 8] / 2 + 128
                 ).astype(np.uint8),
                cv2.COLOR_BGR2GRAY))
        plt.show()
        plt.imshow(
            np_image[scroll_area[1] * 8:(scroll_area[1] + scroll_area[3]) * 8,
                     scroll_area[0] * 8:(scroll_area[0] + scroll_area[2]) * 8])
        plt.show()

    #plt.imshow(np.tile(fullNTs, (2, 2))[sy:sy+scroll_area[3], sx:sx+scroll_area[2]])
    # plt.show()
    # TODO: test this!

    scrolled_nt = nts[sy:sy + scroll_area[3],
                      sx:sx + scroll_area[2]]
    scrolled_attr = attrs[sy:sy + scroll_area[3],
                          sx:sx + scroll_area[2]]
    tm_scroll = (sx, sy)
    px, py = last_tm_scroll if last_tm_scroll is not None else tm_scroll
    mx, my = (sx - px, sy - py)
    if mx >= 16:
        mx -= 32
    if mx <= -16:
        mx += 32
    if my >= 15:
        my -= 30
    if my <= -15:
        my += 30
    tm_motion = (mx, my)
    # print sx, sy, px, py, tm_motion
    return scrolled_nt, scrolled_attr, tm_motion, tm_scroll


def test_sprite_data_(emu, colorized2id, id2colorized, timestep, data):
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
        data[timestep] = (timestep, colorized2id[uniq], sprite)
    return data


def ppu_output(emu, inputVec, **kwargs):
    start = VectorBytes()
   
    emu.save(start)

    peekevery = kwargs.get("peekevery", 1)

    colorized2id = {}
    id2colorized = {}
    tile2colorized = {}
    data = {}
    nametable_outputs = {}
    attr_outputs = {}
    scrolled_nt_outputs = {}
    scrolled_attr_outputs = {}
    xScrolls = None
    motion = {}
    scrolls = {}
    tm_scrolls = {}
    tm_motion = {}
    palettes = {}
    corrs = {}
    img_buffer = VectorBytes()
    img_buffer2 = VectorBytes()
    np_image = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_prev = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
    np_image_temp = None
    debug_output = kwargs.get("debug_output", True)
    get_bg_data = kwargs.get("bg_data", True)
    scroll_area = kwargs.get("scroll_area", (0, 0, 32, 30))
    get_sprite_data = kwargs.get("sprite_data", True)
    get_scroll = kwargs.get("scrolling", True) or get_bg_data
    test_control = kwargs.get("test_control", False)
    inputs2 = kwargs.get("inputs2", [0] * len(inputVec))

    if get_scroll:
        emu.imageInto(img_buffer)
        np_image_prev = convert_image(img_buffer)
        big_picture = np.zeros(shape=(240 * 2, 256 * 2, 3))

    if test_control:

        start_state = fceulib.VectorBytes()
    display = kwargs.get("display", True)
    net_x = 0
    net_y = 0
    offset_left = 8
    offset_top = 8

    has_controls = {}
    for timestep, (inp, inp2) in enumerate(zip(inputVec, inputs2)):
        should_peek = timestep % peekevery == 0
        
        # Have to do this before running this step of input
        if test_control and should_peek:
            has_controls[timestep] = test_control_(
                emu,
                start_state,
                inp, inp2,
                timestep,
                inputVec,
                img_buffer,
                img_buffer2
            )
        
        emu.stepFull(inp, inp2)
        
        if not should_peek:
            continue

        if get_scroll or get_bg_data or test_control:
            
            emu.imageInto(img_buffer)
            
            # TODO: without allocations?
            np_image = convert_image(img_buffer)
            

        
        if get_scroll and timestep > 0:
            mot, scroll, maxv = test_scrolling_visual_(np_image_prev, np_image,
                                                 net_x, net_y,
                                                 offset_left, offset_top,
                                                 scroll_area)
            net_x, net_y = scroll
            motion[timestep] = mot
            scrolls[timestep] = scroll
            corrs[timestep] = maxv

        
        if display:
            outputImage(emu, 'images/{}'.format(timestep), img_buffer)

        
        if get_bg_data:
            nts, attrs, pal = test_bg_data_full_(emu, tile2colorized)
        
            nametable_outputs[timestep] = nts
            attr_outputs[timestep] = attrs
            palettes[timestep] = pal
            (scrolled_nt, scrolled_attr,
             tm_mot, tm_scroll) = test_bg_data_scrolled_(
                nts, attrs, pal, tile2colorized,
                scroll_area,
                timestep,
                tm_scrolls[timestep - peekevery] if timestep > 0 else None,
                 big_picture, np_image,
                 debug_output
            )
            
            scrolled_nt_outputs[timestep] = scrolled_nt
            scrolled_attr_outputs[timestep] = scrolled_attr
            tm_motion[timestep] = tm_mot
            tm_scrolls[timestep] = tm_scroll
        
        if get_sprite_data:
            test_sprite_data_(
                emu,
                colorized2id, id2colorized,
                timestep,
                data)
        np_image_temp = np_image
        np_image = np_image_prev
        np_image_prev = np_image_temp

    emu.load(start)
    results = {}
    if get_scroll:
        results["screen_motion"] = motion
        results["screen_scrolls"] = scrolls
        results["screen_corrs"] = corrs
    if get_bg_data:
        results["full_nametables"] = nametable_outputs
        results["full_attrs"] = attr_outputs
        results["palettes"] = palettes
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


if __name__ == '__main__':
    rom = "metroid.nes"
    #movie = "metroid.fm2"
    start_t = 300
    movie = 'lordtom-metroid-100.fm2'
    #movie = "metroid-long.fm2"

    #rom = 'zelda.nes'
    #movie = 'baxter,jprofit22-legendofzelda.fm2'
    #movie = 'zelda.fm2'

    #rom = 'zelda.nes'
    #movie = 'zelda_dungeon1.fm2'
    #movie = 'zelda.fm2'
    #start_t = 300

    # rom = "smb2u.nes"
    # movie = "smb2u.fm2"
    # start_t = 700
    emu = fceulib.runGame(rom)
    inputs1 = fceulib.readInputs(movie)
    inputs2 = fceulib.readInputs2(movie)

    for i, i2 in zip(inputs1[:start_t], inputs2[:start_t]):
        emu.stepFull(i, i2)

    end = start_t + 3600
    # METROID
    scroll_area = (0, 0, 32, 30 - 0)

    # ZELDA
    #scroll_area= (0,8,32,30-8)

    #MARIO, ZELDA2
    #scroll_area = (0, 4, 32, 30-4)

    ep_data = ppu_output(emu,
                         inputs1[start_t:end],
                         inputs2=inputs2[start_t:end],
                         bg_data=True,
                         scrolling=True,
                         sprite_data=True,
                         colorized_tiles=False,
                         display=False,
                         test_control=True,
                         scroll_area=scroll_area,
                         debug_output=False)
