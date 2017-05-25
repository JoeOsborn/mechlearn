import fceulib
from fceulib import VectorBytes
import numpy as np
import ppu_dump
from fceu_help import pointer_to_numpy, colorize_tile
from fceu_help import get_all_sprites, get_tile, outputImage
import copy
# Needed for Joe's pyenv to find CV2
import site
site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2


class Observation(object):
    __slots__ = (
        "config",
        "inputs", "has_control",
        "sprite_data",
        "nametables", "attrs",
        "full_nametables", "full_attrs",
        "motion", "scrolls", "tm_scrolls", "tm_motion",
        "net_x", "net_y")

    def __init__(self, config, inputs):
        self.config = copy.copy(config)
        self.inputs = [(t, i) for t, i in enumerate(inputs)]
        self.has_control = {}
        self.sprite_data = []
        self.nametables = []
        self.attrs = []
        self.full_nametables = []
        self.full_attrs = []
        self.motion = {}
        self.scrolls = {}
        self.tm_scrolls = {}
        self.tm_motion = {}
        self.net_x = 0
        self.net_y = 0


class MultiTracker(object):
    __slots__ = (
        # config
        "config",
        # stored info
        "colorized2id",
        "id2colorized",
        "tile2colorized",
        "animations",
        # tracks/etc
        "observations",
        # pre-allocated stuff
        "img_buffer",
        "big_picture",
        "np_image",
        "np_image_2",
        "np_image_prev",
        "start"
    )

    def __init__(self, config):
        self.config = {
            "bg_data": True,
            "scroll_area": (0, 0, 32, 30),
            "sprite_data": True,
            "scrolling": True,
            "control": True,
            "display": False,
            "peekevery": 1
        }
        self.config.update(config)
        self.colorized2id = {}
        self.id2colorized = {}
        self.tile2colorized = {}
        self.observations = []
        self.img_buffer = VectorBytes()
        self.big_picture = np.zeros(shape=(240 * 2, 256 * 2, 3))
        self.np_image = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
        self.np_image_2 = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
        self.np_image_prev = np.zeros(shape=(240, 256, 1), dtype=np.uint8)
        self.start = fceulib.VectorBytes()

    def observe(self, emu, inputs, **kwargs):
        config = copy.copy(self.config)
        config.update(kwargs)
        if config.get("scrolling") or config.get("bg_data"):
            emu.imageInto(self.img_buffer)
            self.np_image_prev = ppu_dump.convert_image(self.img_buffer)

        o = Observation(config, inputs)
        for timestep, inp in enumerate(inputs):
            self.extend_observation(emu, config, o, timestep, inp, inputs)
        return o

    def extend_observation(self, emu, config, o, timestep, inp, inputs):
        offset_left = 8
        offset_top = 8
        scroll_window = 5

        peekevery = config.get("peekevery")
        display = config.get("display")
        get_bg_data = config.get("bg_data")
        scroll_area = config.get("scroll_area")
        get_sprite_data = config.get("sprite_data")
        get_scroll = config.get("scrolling") or get_bg_data
        test_control = config.get("control")

        np_image_temp = None
        if not (timestep % peekevery == 0):
            return
        if test_control:
            emu.save(self.start)
            emu.stepFull(inp, 0x0)
            next = timestep + 1
            if next >= len(inputs):
                next = len(inputs) - 1
            emu.stepFull(inputs[next], 0x0)
            emu.imageInto(self.img_buffer)
            self.np_image = ppu_dump.convert_image(self.img_buffer)
            has_control = False
            for test_inp in [0, 1, 4, 5, 6, 7]:
                emu.load(self.start)
                emu.stepFull(1 << test_inp, 0x0)
                emu.stepFull(inputs[next], 0x0)
                emu.imageInto(self.img_buffer)
                self.np_image_2 = ppu_dump.convert_image(self.img_buffer)
                if np.sum(np.abs(self.np_image_2 - self.np_image)) > 0:
                    has_control = True
                    break
            o.has_control[timestep] = has_control
            emu.load(self.start)
        emu.stepFull(inp, 0x0)
        if get_scroll or get_bg_data or test_control:
            emu.imageInto(self.img_buffer)
            # TODO: without allocations?
            self.np_image = ppu_dump.convert_image(self.img_buffer)
        if get_scroll and timestep > 0:
            # TODO: maybe instead consider a span of columns on the left
            # and middle and right and a span of rows on the top and middle
            # and bottom, and see which of those are moving in what
            # direction, and take the biggest/average scroll?
            result = cv2.matchTemplate(
                self.np_image_prev[scroll_area[1] * 8:
                                   (scroll_area[1] + scroll_area[3]) * 8,
                                   scroll_area[0] * 8:
                                   (scroll_area[0] + scroll_area[2]) * 8],
                self.np_image[scroll_area[1] * 8 +
                              offset_top:scroll_area[1] * 8 +
                              scroll_area[3] * 8 - offset_top * 2,
                              scroll_area[0] * 8 + offset_left:
                              scroll_area[0] * 8 +
                              scroll_area[2] * 8 - offset_left * 2],
                cv2.TM_CCOEFF_NORMED
            )
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(result)
            print "Match1", minv, maxv, minloc, maxloc
            best_sx, best_sy = 0, 0
            cx, cy = offset_left, offset_top
            best_match = result[cy, cx]
            # Look around the center of the image.
            # does it get better-matched
            # going to the left, right, up, or down?
            for sx in range(-scroll_window, scroll_window):
                for sy in range(-scroll_window, scroll_window):
                    match = result[cy + sy, cx + sx]
                    if match > best_match:
                        best_sx = sx
                        best_sy = sy
                        best_match = match
            o.net_x += best_sx
            o.net_y += best_sy
            print "Sc1 Offset:", best_sx, best_sy, o.net_x, o.net_y
            o.motion[timestep] = (best_sx, best_sy)
            o.scrolls[timestep] = (o.net_x, o.net_y)
        if display:
            outputImage(emu, 'images/{}'.format(timestep), self.img_buffer)
        if get_bg_data:
            h_neighbors = {0: 1, 1: 0, 2: 3, 3: 2}
            v_neighbors = {0: 2, 2: 0, 1: 3, 3: 1}

            base_nti = emu.fc.ppu.values[0] & 0x3
            right_nti = h_neighbors[base_nti]
            below_nti = v_neighbors[base_nti]
            right_below_nti = v_neighbors[right_nti]

            print ("NTS:\n",
                   base_nti, right_nti, "\n",
                   below_nti, right_below_nti)

            # nt
            # Getting the mirroring right and grabbing the right tile seems
            # done by the PPUTile function in fceulib's ppu.cc.
            # But it has lots of parameters including ones
            #  related to MMC5HackMode
            #  and other mapper stuff.  Because mappers are determined by
            #  mirroring!
            #  But it also relies on the global Pline to figure out which
            #  row it's in...
            #  and the `scanline` global...
            #
            nta = pointer_to_numpy(emu.fc.ppu.NTARAM, copy=False)
            # change to handle other nametables?
            mirroring = emu.fc.cart.mirroring
            print "M", mirroring, "base", base_nti
            # 0 - Hori
            # 1 - Vert
            # 2 - all use 0
            # 3 - all use 1

            # Later, just look at VNAPages to handle roms with mappers with
            # extra vram?
            base_nt, base_attr = ppu_dump.nt_page(nta, base_nti, mirroring)
            right_nt, right_attr = ppu_dump.nt_page(nta, right_nti, mirroring)
            below_nt, below_attr = ppu_dump.nt_page(nta, below_nti, mirroring)
            right_below_nt, right_below_attr = ppu_dump.nt_page(
                nta,
                right_below_nti,
                mirroring)

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

            o.full_nametables.append(fullNTs)
            o.full_attrs.append(fullAttr)

            pairs = set()
            pt = pointer_to_numpy(emu.fc.ppu.PALRAM, copy=False)
            for ii in range(fullAttr.shape[0]):
                for jj in range(fullAttr.shape[1]):
                    pairs.add((int(fullNTs[ii, jj]),
                               int(fullAttr[ii, jj])))
            for pair in pairs:
                if pair not in self.tile2colorized:
                    self.tile2colorized[pair] = colorize_tile(
                        get_tile(pair[0], emu.fc),
                        pair[1],
                        pt)[:, :, :3]
                    # Have to divide by 255 to actually
                    # display with plt.imshow

            # OK, let's figure out our scrolling situation.  First
            # build a mighty template image out of the whole picture.
            # We're gonna template match to see how the baby real
            # image fits inside the big full screen image.  We can't
            # really use the scroll info determined earlier, because
            # we don't know the x and y scroll as of initialization
            # time.
            for ii in range(fullAttr.shape[0]):
                for jj in range(fullAttr.shape[1]):
                    pair = (int(fullNTs[ii, jj]),
                            int(fullAttr[ii, jj]))
                    self.big_picture[ii * 8:ii * 8 + 8,
                                     jj * 8:jj * 8 + 8,
                                     :] = self.tile2colorized[pair] / 255.0

            insets = cv2.matchTemplate(
                self.big_picture.astype(np.uint8),
                self.np_image[
                    scroll_area[1] * 8:(scroll_area[1] + scroll_area[3]) * 8,
                    scroll_area[0] * 8:(scroll_area[0] + scroll_area[2]) * 8],
                cv2.TM_CCOEFF_NORMED
            )
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(insets)
            sx = maxloc[0] / 8
            sy = maxloc[1] / 8
            print "Sc2:", sx, sy
            # TODO: test this!
            o.nametables.append(fullNTs[sy:sy + scroll_area[3],
                                        sx:sx + scroll_area[2]])
            o.attrs.append(fullAttr[sy:sy + scroll_area[3],
                                    sx:sx + scroll_area[2]])
            o.tm_scrolls[timestep] = (sx, sy)
            px, py = o.tm_scrolls[timestep - 1] if timestep > 0 else (sx, sy)
            mx, my = (sx - px, sy - py)
            if mx >= 16:
                mx -= 32
            if mx <= -16:
                mx += 32
            if my >= 15:
                my -= 30
            if my <= -15:
                my += 30
            o.tm_motion[timestep] = (mx, my)
            print sx, sy, px, py, o.tm_motion[timestep]
        if get_sprite_data:
            sprite_list, colorized_sprites = get_all_sprites(emu.fc)
            for sprite_id, sprite in enumerate(sprite_list):
                if np.sum(colorized_sprites[sprite_id].ravel()) == 0:
                    continue
                uniq = tuple(colorized_sprites[sprite_id].ravel())
                if uniq not in self.colorized2id:
                    self.colorized2id[uniq] = len(self.colorized2id)
                    # plt.imshow(colorized_sprites[sprite_id][:,:,:3]/255.)
                    # plt.show()
                    self.id2colorized[
                        self.colorized2id[uniq]
                    ] = colorized_sprites[sprite_id]
                # print timestep,  colorized2id[uniq], sprite[:2]
                o.sprite_data.append((timestep,
                                      self.colorized2id[uniq],
                                      sprite))
        np_image_temp = self.np_image
        self.np_image = self.np_image_prev
        self.np_image_prev = np_image_temp
