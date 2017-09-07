# -*- default-directory: "~/Projects/mechlearn/fami/"; compile-command: "python -m nes.tiles " -*-

import zmq
import numpy
import matplotlib.pyplot as plt
import palette

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
    start = mirror_modes[mirroring][nti] * 1024
    base_nt = nt[start: start + 960]
    base_attr = nt[start + 960: start + 960 + 64]
    nt = base_nt.reshape(30, 32)
    attr = base_attr.reshape(8, 8)
    full_attr = numpy.zeros([16, 16], dtype=numpy.uint8)
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
    full_attr = numpy.kron(full_attr, numpy.ones((2, 2), dtype=numpy.uint8))
    return nt, full_attr


def get_tile_pattern(tileid, ppuv, chrram):
    bg_pat_addr = 0x1000 if (ppuv[0] & (1 << 4)) else 0x0000
    print tileid, bg_pat_addr, bg_pat_addr + tileid * 16
    tile = numpy.zeros(shape=(8, 8), dtype=numpy.uint8)
    for yy in range(tile.shape[0]):
        lb = chrram[bg_pat_addr + tileid * 16 + yy]
        ub = chrram[bg_pat_addr + tileid * 16 + yy + 8]
        for xx in range(tile.shape[1]):
            l = lb & 1
            u = ub & 1
            v = l + u * 2
            lb >>= 1
            ub >>= 1
            tile[yy, tile.shape[1] - xx - 1] = v
    return tile


def get_colors(tileatt, ppuv, pal):
    offset = 0
    palette_id = offset | (tileatt << 2)
    result = numpy.zeros(shape=(4, 4))
    # TODO: consider that tiles may have transparency and sprites can draw behind them!
    # This means that we can't colorize or at least can't draw the big picture
    # until we know if a sprite pixel is overlapping a tile pixel and that
    # sprite has background priority.
    for i in range(0, 4):
        color_id = pal[palette_id | i]
        result[i, :3] = palette.ntsc[color_id * 3:color_id * 3 + 3]
        result[i, 3] = 255.
    return result


def colorize(pat, cols):
    colored = numpy.zeros(shape=(pat.shape[0], pat.shape[1], 4))
    for ii in range(pat.shape[0]):
        for jj in range(pat.shape[1]):
            colored[ii, jj] = cols[pat[ii, jj]]
    return colored / 255.


def start(services):
    context = zmq.Context()
    emusocket = context.socket(zmq.REQ)
    emusocket.connect("tcp://127.0.0.1:" + services["fceux"])
    ctrlsocket = context.socket(zmq.REP)
    ctrlsocket.bind("tcp://127.0.0.1:" + services["tiles"])

    # Wait for requests like "what are the active tilemaps at a given time,
    # and how do those tile IDs map to colorized tiles"?  This yields all four
    # NTAs at once, always in the same order.  If you want to do template
    # matching you should probably repeat/wrap at the edges.
    while True:
        msg = ctrlsocket.recv_json()
        print "Got msg", msg
        state = msg["state"]
        emusocket.send_json({
            "state": state,
            "data": ["nta", "mirror", "chr", "pal", "ppu"]
        })
        # get back results and yield:
        full_result = emusocket.recv_json()
        result = full_result["data"][-1]
        mirroring = result["mirror"]
        nta = numpy.array(result["nta"], dtype=numpy.uint8)
        nt00, attr00 = nt_page(nta, 0, mirroring)
        nt10, attr10 = nt_page(nta, 1, mirroring)
        nt01, attr01 = nt_page(nta, 2, mirroring)
        nt11, attr11 = nt_page(nta, 3, mirroring)

        fullNTs = numpy.vstack([
            numpy.hstack([
                nt00,
                nt10
            ]),
            numpy.hstack([
                nt01,
                nt11
            ])
        ])
        fullAttr = numpy.vstack([
            numpy.hstack([
                attr00,
                attr10
            ]),
            numpy.hstack([
                attr01,
                attr11
            ])
        ])

        tiles2patterns = {}
        attr2colors = {}

        pal = numpy.array(result["pal"], dtype=numpy.uint8)
        ppuv = result["ppu"]
        ppumask = ppuv[1]
        chrram = numpy.array(result["chr"], dtype=numpy.uint8)
        big_picture = numpy.zeros(shape=(fullAttr.shape[0] * 8,
                                         fullAttr.shape[1] * 8,
                                         4),
                                  dtype=numpy.float32)
        color_tiles = {}
        rendered = ppumask & (1 << 3)
        for ii in range(fullAttr.shape[0]):
            for jj in range(fullAttr.shape[1]):
                tileid = fullNTs[ii, jj]
                if tileid in tiles2patterns:
                    pat = tiles2patterns[tileid]
                else:
                    pat = get_tile_pattern(tileid, ppuv, chrram)
                    tiles2patterns[tileid] = pat
                tileatt = fullAttr[ii, jj]
                if tileatt in attr2colors:
                    tilecols = attr2colors[tileatt]
                else:
                    tilecols = get_colors(tileatt, ppuv, pal)
                    attr2colors[tileatt] = tilecols
                # TODO: store as two level dict
                if tileid not in color_tiles:
                    color_tiles[tileid] = {}
                if tileatt not in color_tiles[tileid]:
                    color_tiles[tileid][tileatt] = colorize(pat, tilecols)
                if rendered:
                    big_picture[ii * 8:ii * 8 + 8,
                                jj * 8:jj * 8 + 8,
                                :] = color_tiles[tileid][tileatt]
        output = {"state": state,
                  "rendered": rendered}
        for d in msg["data"]:
            if d == "tilemap":
                output[d] = fullNTs.tolist()
            elif d == "tilepals":
                output[d] = fullAttr.tolist()
            elif d == "tilemap_pixels":
                output[d] = big_picture.tolist()
            elif d == "palette":
                output[d] = pal.tolist()
            elif d == "tiles_to_patterns":
                output[d] = dict(map(
                    lambda (k, v): (str(k), v.tolist()),
                    tiles2patterns.items()))
            elif d == "pals_to_colors":
                output[d] = dict(map(
                    lambda (k, v): (str(k), v.tolist()),
                    attr2colors.items()))
            elif d == "color_tiles":
                # TODO: store or at least output as two-level dict because of
                # compound key
                output[d] = dict(map(
                    lambda (k, v): (str(k),
                                    dict(map(lambda (k2, v2): (str(k2),
                                                               v2.tolist()),
                                             v.items()))),
                    color_tiles.items()))
            else:
                assert False, "Unrecognized data request" + str(d)
        ctrlsocket.send_json(output)


if __name__ == "__main__":
    import sys
    import json
    start(json.load(open("services.json", 'r')
                    if len(sys.argv) < 2 else sys.argv[1]))
