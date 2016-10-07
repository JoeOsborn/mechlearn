import fceulib
from fceulib import VectorBytes
from PIL import Image
import random

def go():
    emu = fceulib.runGame("mario.nes")
    inputVec = fceulib.readInputs('happylee-supermariobros,warped.fm2')
    
    for kk in range(len(inputVec)):
        emu.stepFull(inputVec[kk], 0x0)
        print(kk)
        if kk % 300 == 0:
            outBytes = VectorBytes()
            print 'printing'
            emu.imageInto(outBytes)
            outBytes = [outBytes[ii] for ii in range(len(outBytes))]
            outImg = Image.frombytes("RGBA", (256, 256), str(bytearray(outBytes)))
            outImg.save("run{}.png".format(kk))


go()
