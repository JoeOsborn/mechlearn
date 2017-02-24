import fceulib
from fceulib import VectorBytes
import numpy as np
from fceu_help import pointer_to_numpy, colorize_tile
from fceu_help import get_all_sprites, get_tile, get_sprite,outputImage
from math import log
import matplotlib.pyplot as plt
# Needed for Joe's pyenv to find CV2
import site
site.addsitedir("/usr/local/lib/python2.7/site-packages")


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
    get_bg_data = kwargs.get("bg_data", True)
    get_colorized_tiles = kwargs.get("colorized_tiles", True)
    get_sprite_data = kwargs.get("sprite_data", True)

    display =  kwargs.get("display", True)
    for timestep, inp in enumerate(inputVec):
        emu.stepFull(inp, 0x0)
        if not (timestep % peekevery == 0):
            continue
        if xScrolls is not None:
            xScrolls[timestep] = emu.xScroll

        xScroll = emu.fc.ppu.xScroll
        yScroll = emu.fc.ppu.yScroll
        fineXScroll = xScroll & 0x7
        coarseXScroll = xScroll >> 3

        if display:
            outputImage(emu, 'images/{}'.format(timestep),img_buffer)
            

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
                    #plt.imshow(colorized_sprites[sprite_id][:,:,:3]/255.)
                    #plt.show()
                    id2colorized[colorized2id[uniq]] = colorized_sprites[sprite_id]
                #print timestep,  colorized2id[uniq], sprite[:2]
                data.append((timestep, colorized2id[uniq], sprite))

    emu.load(start)
    results = {
        "screen_motion": motion,
    }
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
