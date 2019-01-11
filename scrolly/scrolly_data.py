import remocon
import itertools
from PIL import Image

# read playfile
fm2file = "lordtom_tompa-smb3-100.fm2"
info = remocon.read_fm2("../plays/" + fm2file)
romname = info["romFilename"][0] + ".nes"
controls_fm = info["player_controls"]
controls = list(map(remocon.moves_to_bytes, controls_fm))
tmax = min(len(controls[0]),510)

remo = remocon.mesen(
    "../mesen/remocon/obj.x64/remocon",
    "../roms/"+romname)

tskip = 500+(hash(fm2file) % 100)
remo.step([controls[0][0:tskip-1]], remocon.Infos())
tsteps = [11,5,7,3,15,1,13]
stepper = itertools.cycle(tsteps)
t = tskip
frame0 = remo.step([controls[0][tskip-1:tskip]], remocon.Infos(framebuffer=True,tiles_by_pixel=True))[0][0]
fb0 = frame0.framebuffer
scroll0 = scroll_mode(frame0)
scrolls = [(t,0)]
while t < tmax:
    t0 = t
    t += tsteps.next()
    if tstep > 1:
        remo.step([controls[0][t0:t-1]], remocon.Infos())
    frame = remo.step([controls[0][t-1:t]], remocon.Infos(framebuffer=True,tiles_by_pixel=True))[0][0]
    fb = frame.framebuffer
    scroll = scroll_mode(frame)

    # write out fb for t
    Image.fromarray(fb).save("../derived/{}/{}/t_{}.png",romname,fm2file,t)
    # add t, scrolldelta to csv
    scrolls.push((t,scroll-scroll0))
    
    fb0 = fb
    scroll0 = scroll
    
with open("../derived/{}/{}/scrolls.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(scrolls)
