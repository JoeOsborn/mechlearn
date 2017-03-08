import fceulib
import ppu_dump
import tracking
import sys

if __name__ == "__main__":
    rom = sys.argv[1]
    movie = sys.argv[2]
    start_t = int(sys.argv[3])
    outputname = sys.argv[4]
    emu = fceulib.runGame(rom)
    inputs = fceulib.readInputs(movie)
    for i in inputs[:start_t]:
        emu.stepFull(i, 0x0)
    ep_data = ppu_dump.ppu_output(emu,
                                  inputs[start_t:],
                                  bg_data=False,
                                  scrolling=True,
                                  sprite_data=False,
                                  display=False)
    # (ep_tracks, old_ep_tracks) = tracking.tracks_from_sprite_data(
    #     ep_data["sprite_data"]
    # )
    
