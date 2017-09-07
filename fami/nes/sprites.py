# -*- default-directory: "~/Projects/mechlearn/fami/"; compile-command: "python -m nes.sprites " -*-

import zmq
import numpy
import palette


def get_colors(tileatt, ppuv, pal, is_sprite=True):
    offset = 1 << 4 if is_sprite else 0
    palette_id = offset + (tileatt << 2)
    result = numpy.zeros(shape=(4, 4))
    for i in range(1, 4):
        color_id = pal[palette_id + i]
        result[i, :3] = palette.ntsc[color_id * 3:color_id * 3 + 3]
        result[i, 3] = 255.
    return result


def colorize(pat, cols, hflip, vflip):
    output = numpy.zeros(shape=(pat.shape[0], pat.shape[1], 4))
    for ii in range(pat.shape[0]):
        for jj in range(pat.shape[1]):
            output[ii, jj] = cols[pat[ii, jj]]
    if hflip:
        output = output[:, ::-1, :]
    if vflip:
        output = output[::-1, :, :]

    return output / 255.


masks = {'is_background': 0x20,
         'hflip': 0x40,
         'vflip': 0x80}


def get_sprite_pattern(tile_id, ppuv, chrram):
    repeat = 1

    if (ppuv[0] & (1 << 5)):
        repeat = 2
        fg_pat_addr = 0x1000 if tile_id & 0x01 else 0x0000
        fg_ram = chrram[fg_pat_addr:fg_pat_addr + 0x1000]
        tile_id = (tile_id >> 1) << 1
    else:
        fg_pat_addr = 0x1000 if (ppuv[0] & (1 << 3)) else 0x0000
        fg_ram = chrram[fg_pat_addr:fg_pat_addr + 0x1000]
    tile = numpy.zeros(shape=(8 if repeat == 1 else 16, 8),
                       dtype=numpy.uint8)
    for ii in range(repeat):
        for yy in range(8):
            lb = fg_ram[(tile_id + ii) * 16 + yy]
            ub = fg_ram[(tile_id + ii) * 16 + yy + 8]
            for xx in range(8):
                l = lb & 1
                u = ub & 1
                v = l + u * 2
                lb >>= 1
                ub >>= 1
                tile[ii * 8 + yy, 8 - xx - 1] = v
    return tile


def start(services):
    context = zmq.Context()
    emusocket = context.socket(zmq.REQ)
    emusocket.connect("tcp://127.0.0.1:" + services["fceux"])
    ctrlsocket = context.socket(zmq.REP)
    ctrlsocket.bind("tcp://127.0.0.1:" + services["sprites"])

    # Wait for requests like "what are the active sprites at a given time,
    # and how do those map to colorized sprites"?
    while True:
        msg = ctrlsocket.recv_json()
        print "Got msg", msg
        state = msg["state"]
        emusocket.send_json({
            "state": state,
            "data": ["oam", "chr", "pal", "ppu"]
        })
        # get back results and yield:
        full_result = emusocket.recv_json()
        result = full_result["data"][-1]

        pal = numpy.array(result["pal"], dtype=numpy.uint8)
        ppuv = result["ppu"]
        ppumask = ppuv[1]
        oam = numpy.array(result["oam"], dtype=numpy.uint8)
        sprites = []
        chrram = numpy.array(result["chr"], dtype=numpy.uint8)
        sprites2patterns = {}
        attr2colors = {}
        colored_sprites = {}
        rendered = ppumask & (1 << 4)
        for spri in range(0, len(oam), 4):
            if oam[spri] >= 240:
                sprites.append(None)
                continue
            attrs = oam[spri + 2].item()
            attrib = {mask: attrs & masks[mask] > 0 for mask in masks}
            pal_id = attrs & 0x03
            attrib['palette_id'] = pal_id
            attrib['x'] = oam[spri + 3].item() * 1.
            attrib['y'] = oam[spri].item() * 1. + 1
            pat_id = oam[spri + 1].item()
            attrib['pattern_id'] = pat_id
            sprites.append(attrib)
            pat = get_sprite_pattern(pat_id,
                                     ppuv,
                                     chrram)
            sprites2patterns[pat_id] = pat
            if pal_id in attr2colors:
                spritecols = attr2colors[pal_id]
            else:
                spritecols = get_colors(pal_id, ppuv, pal)
                attr2colors[pal_id] = spritecols
            # TODO: store as two level dict
            if pat_id not in colored_sprites:
                colored_sprites[pat_id] = {}
            if attrs not in colored_sprites[pat_id]:
                colored_sprites[pat_id][attrs] = colorize(
                    pat, spritecols,
                    attrib["hflip"], attrib["vflip"])
        output = {"state": state,
                  "rendered": rendered}
        for d in msg["data"]:
            if d == "sprites":
                output[d] = sprites
            elif d == "palette":
                output[d] = pal.tolist()
            elif d == "sprites_to_patterns":
                output[d] = dict(map(
                    lambda (k, v): (str(k), v.tolist()),
                    sprites2patterns.items()))
            elif d == "pals_to_colors":
                output[d] = dict(map(
                    lambda (k, v): (str(k), v.tolist()),
                    attr2colors.items()))
            elif d == "color_sprites":
                output[d] = dict(map(
                    lambda (k, v): (str(k),
                                    dict(map(lambda (k2, v2): (str(k2),
                                                               v2.tolist()),
                                             v.items()))),
                    colored_sprites.items()))
            else:
                assert False, "Unrecognized data request" + str(d)
        ctrlsocket.send_json(output)


if __name__ == "__main__":
    import sys
    import json
    start(json.load(open("services.json", 'r')
                    if len(sys.argv) < 2 else sys.argv[1]))
