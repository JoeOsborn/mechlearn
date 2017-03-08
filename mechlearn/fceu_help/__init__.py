import numpy as np
from PIL import Image


def pointer_to_numpy(ptr, length=0):
    if length == 0:
        length = len(ptr)
    return np.array([ptr[xx] for xx in range(length)])


def hold(mask, duration):
    return [mask for i in range(duration)]


def outputImage(emu, name, buf):
    emu.imageInto(buf)
    outImg = Image.frombytes("RGBA", (256, 240), str(bytearray(buf)))
    outImg.save(name + ".png")


def sprite_attributes_to_dict(attributes):
    masks = {'background': 0x20,
             'hflip': 0x40,
             'vflip': 0x80}
    sprite_attributes = {mask: attributes & masks[mask] > 0 for mask in masks}
    sprite_attributes['palette'] = attributes & 0x03
    sprite_attributes['table'] = attributes & 0x01
    return sprite_attributes


def get_sprite(tile_id, table, fc):
    repeat = 1

    if (fc.ppu.values[0] & (1 << 5)):
        repeat = 2
        fg_pat_addr = 0x1000 if table else 0x0000
        #fg_pat_addr = 0x1000 if (tile_id & 0x01) else 0x0000

        fg_ram = pointer_to_numpy(fc.cart.getVPageChunk(fg_pat_addr), 0x1000)
        tile_id = (tile_id >> 1) << 1
    else:
        fg_pat_addr = 0x0000 if (fc.ppu.values[0] & (1 << 4)) else 0x1000
        fg_ram = pointer_to_numpy(fc.cart.getVPageChunk(fg_pat_addr), 0x1000)

    tile = []
    for ii in range(repeat):
        for yy in range(8):
            row = []
            lb = fg_ram[(tile_id + ii) * 16 + yy]
            ub = fg_ram[(tile_id + ii) * 16 + yy + 8]
            for xx in range(8):
                l = lb & 1
                u = ub & 1
                v = l + u * 2
                lb >>= 1
                ub >>= 1
                row.append(v)

            tile.append(list(reversed(row)))
    tile = np.array(tile)
    return tile


def get_tile(tile_id, fc):

    bg_pat_addr = 0x1000 if (fc.ppu.values[0] & (1 << 4)) else 0x0000
    bg_ram = pointer_to_numpy(fc.cart.getVPageChunk(bg_pat_addr), 0x1000)

    tile = []
    for yy in range(8):
        row = []
        lb = bg_ram[tile_id * 16 + yy]
        ub = bg_ram[tile_id * 16 + yy + 8]
        for xx in range(8):
            l = lb & 1
            u = ub & 1
            v = l + u * 2
            lb >>= 1
            ub >>= 1
            row.append(v)
        tile.append(list(reversed(row)))
    tile = np.array(tile)
    return tile


def colorize_tile(tile, attribute, palette_table):
    ntsc_palette = [0x80, 0x80, 0x80, 0x00, 0x3D, 0xA6, 0x00, 0x12, 0xB0, 0x44, 0x00, 0x96,
                    0xA1, 0x00, 0x5E, 0xC7, 0x00, 0x28, 0xBA, 0x06, 0x00, 0x8C, 0x17, 0x00,
                    0x5C, 0x2F, 0x00, 0x10, 0x45, 0x00, 0x05, 0x4A, 0x00, 0x00, 0x47, 0x2E,
                    0x00, 0x41, 0x66, 0x00, 0x00, 0x00, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
                    0xC7, 0xC7, 0xC7, 0x00, 0x77, 0xFF, 0x21, 0x55, 0xFF, 0x82, 0x37, 0xFA,
                    0xEB, 0x2F, 0xB5, 0xFF, 0x29, 0x50, 0xFF, 0x22, 0x00, 0xD6, 0x32, 0x00,
                    0xC4, 0x62, 0x00, 0x35, 0x80, 0x00, 0x05, 0x8F, 0x00, 0x00, 0x8A, 0x55,
                    0x00, 0x99, 0xCC, 0x21, 0x21, 0x21, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
                    0xFF, 0xFF, 0xFF, 0x0F, 0xD7, 0xFF, 0x69, 0xA2, 0xFF, 0xD4, 0x80, 0xFF,
                    0xFF, 0x45, 0xF3, 0xFF, 0x61, 0x8B, 0xFF, 0x88, 0x33, 0xFF, 0x9C, 0x12,
                    0xFA, 0xBC, 0x20, 0x9F, 0xE3, 0x0E, 0x2B, 0xF0, 0x35, 0x0C, 0xF0, 0xA4,
                    0x05, 0xFB, 0xFF, 0x5E, 0x5E, 0x5E, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
                    0xFF, 0xFF, 0xFF, 0xA6, 0xFC, 0xFF, 0xB3, 0xEC, 0xFF, 0xDA, 0xAB, 0xEB,
                    0xFF, 0xA8, 0xF9, 0xFF, 0xAB, 0xB3, 0xFF, 0xD2, 0xB0, 0xFF, 0xEF, 0xA6,
                    0xFF, 0xF7, 0x9C, 0xD7, 0xE8, 0x95, 0xA6, 0xED, 0xAF, 0xA2, 0xF2, 0xDA,
                    0x99, 0xFF, 0xFC, 0xDD, 0xDD, 0xDD, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11]
    output = np.zeros([tile.shape[0], tile.shape[1], 4])
    for yy in range(tile.shape[0]):
        for xx in range(tile.shape[1]):
            palette_id = (attribute << 2 | tile[yy, xx])
            color_id = palette_table[palette_id]
            output[yy, xx, 0] = ntsc_palette[color_id * 3 + 0]
            output[yy, xx, 1] = ntsc_palette[color_id * 3 + 1]
            output[yy, xx, 2] = ntsc_palette[color_id * 3 + 2]
            output[yy, xx, 3] = 0xFF
    return output


def get_all_sprites(fc):
    ram_ptr = fc.fceu.RAM
    pt = pointer_to_numpy(fc.ppu.PALRAM)
    sprite_ram = pointer_to_numpy(fc.ppu.SPRAM)
    output = []
    ids = set()
    attributes = []
    for ii in range(0, len(sprite_ram), 4):
        if sprite_ram[ii] < 240:
            attr = sprite_attributes_to_dict(sprite_ram[ii + 2])
            attributes.append(attr)
            # x, y, idx, bg, pal, hflip, vlip, ppu_regs
            ppu_values = fc.ppu.values
            output.append([sprite_ram[ii + 3] * 1., sprite_ram[ii] * 1. + 1, sprite_ram[ii + 1],
                           attr['background'], attr['palette'], attr[
                               'hflip'], attr['vflip'],
                           [ppu_values[0],
                            ppu_values[1],
                            ppu_values[2],
                            ppu_values[3]]])
            ids.add((sprite_ram[ii + 1], attr['table']))
    sprite_map = {id[0]: get_sprite(id[0], id[1], fc) for id in ids}
    colorized_sprites = []
    for sprite, attr in zip(output, attributes):
        sprite_pixels = sprite_map[sprite[2]]
        colorized = colorize_sprite(sprite_pixels, pt, attr)
        colorized_sprites.append(colorized)
    return output, colorized_sprites


def colorize_sprite(sprite, palette_table, attributes):
    ntsc_palette = [0x80, 0x80, 0x80, 0x00, 0x3D, 0xA6, 0x00, 0x12, 0xB0, 0x44, 0x00, 0x96,
                    0xA1, 0x00, 0x5E, 0xC7, 0x00, 0x28, 0xBA, 0x06, 0x00, 0x8C, 0x17, 0x00,
                    0x5C, 0x2F, 0x00, 0x10, 0x45, 0x00, 0x05, 0x4A, 0x00, 0x00, 0x47, 0x2E,
                    0x00, 0x41, 0x66, 0x00, 0x00, 0x00, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
                    0xC7, 0xC7, 0xC7, 0x00, 0x77, 0xFF, 0x21, 0x55, 0xFF, 0x82, 0x37, 0xFA,
                    0xEB, 0x2F, 0xB5, 0xFF, 0x29, 0x50, 0xFF, 0x22, 0x00, 0xD6, 0x32, 0x00,
                    0xC4, 0x62, 0x00, 0x35, 0x80, 0x00, 0x05, 0x8F, 0x00, 0x00, 0x8A, 0x55,
                    0x00, 0x99, 0xCC, 0x21, 0x21, 0x21, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
                    0xFF, 0xFF, 0xFF, 0x0F, 0xD7, 0xFF, 0x69, 0xA2, 0xFF, 0xD4, 0x80, 0xFF,
                    0xFF, 0x45, 0xF3, 0xFF, 0x61, 0x8B, 0xFF, 0x88, 0x33, 0xFF, 0x9C, 0x12,
                    0xFA, 0xBC, 0x20, 0x9F, 0xE3, 0x0E, 0x2B, 0xF0, 0x35, 0x0C, 0xF0, 0xA4,
                    0x05, 0xFB, 0xFF, 0x5E, 0x5E, 0x5E, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D,
                    0xFF, 0xFF, 0xFF, 0xA6, 0xFC, 0xFF, 0xB3, 0xEC, 0xFF, 0xDA, 0xAB, 0xEB,
                    0xFF, 0xA8, 0xF9, 0xFF, 0xAB, 0xB3, 0xFF, 0xD2, 0xB0, 0xFF, 0xEF, 0xA6,
                    0xFF, 0xF7, 0x9C, 0xD7, 0xE8, 0x95, 0xA6, 0xED, 0xAF, 0xA2, 0xF2, 0xDA,
                    0x99, 0xFF, 0xFC, 0xDD, 0xDD, 0xDD, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11]
    output = np.zeros([sprite.shape[0], sprite.shape[1], 4])
    for yy in range(sprite.shape[0]):
        for xx in range(sprite.shape[1]):
            if sprite[yy, xx] != 0:
                palette_id = 0x10 + \
                    ((attributes['palette'] << 2) | sprite[yy, xx])
                color_id = palette_table[palette_id]
                output[yy, xx, 0] = ntsc_palette[color_id * 3 + 0]
                output[yy, xx, 1] = ntsc_palette[color_id * 3 + 1]
                output[yy, xx, 2] = ntsc_palette[color_id * 3 + 2]
                output[yy, xx, 3] = 0xFF
    if attributes['hflip']:
        output = output[:, ::-1, :]
    if attributes['vflip']:
        output = output[::-1, :, :]

    return output
