import fceulib
from fceulib import VectorBytes
from PIL import Image

START = 0x08
RUN = 0x02
JUMP = 0x01

mario_x = 0x006D
mario_y = 0x00CE

imgBuffer = VectorBytes()


def hold(mask, duration):
    return [mask for i in range(duration)]


def outputImage(emu, name, buf=imgBuffer):
    emu.imageInto(buf)
    outImg = Image.frombytes("RGBA", (256, 256), str(bytearray(buf)))
    outImg.save(name + ".png")


J_START = 0
J_RISE = 1
J_FALL = 2
J_END = 3


def go(game):
    total = 0
    emu = fceulib.runGame(game)
    startInputs = hold(0x0, 120) + hold(START | JUMP, 30) + hold(0x0, 150)
    jumpInputs = [hold(0x0, 5) + hold(JUMP, t) + hold(0x0, 300)
                  for t in range(5, 300)]
    for m in startInputs:
        total = total + 1
        emu.step(m, 0x0)
    total = total + 1
    emu.stepFull(0x0, 0x0)
    start = VectorBytes()
    print("SAVE")
    emu.save(start)
    print("SAVED")
    outputImage(emu, "start")

    # for now: hard-code mario x y pointers, but derive velocity numerically
    # we want to note:
    #  starting y position
    #  when the jump starts
    #  initial jump speed and acceleration due to gravity while rising (
    #    might be inconsistent as the \"additive jump force threshold\" is
    #    reached)
    #  when the apex is reached (time and y position)
    #  acceleration due to gravity while falling (we can find
    #    \"additive jump force\" by the difference between falling
    #    and rising gravity)
    #  when we land again
    # Fasterholdt/Pichlmair/Holmgard features:
    # Jump
    #  Gravity u/s2
    #  Terminal Velocity u/s
    #  Takeoff Velocity Vertical u/s
    #  Mario has a tiered jump takeoff velocity (13.75, 14.25, 14.75, 15.75)
    #    based on run speed (0-4, 4-8, 8-12, over 12 u/s) at takeoff
    #  Takeoff Velocity Horizontal u/s
    #  Maximum Jump Height u (derived)
    #  Maximum Jump Distance u (derived)
    #  Maximum Jump Duration s
    # Landing
    #  Instant Break (vy=0 when jump released)
    #  Minimum Jump Duration s (mario capping-upwards-velocity-on-release
    #    behavior)
    #  Additive Jump Force u/s2
    #  Additive Jump Force, Threshold u/s2 Release Drag u/s2
    #  Hold Jump Input (this refers to holding horizontal while jumping)
    #  Minimum Jump Height u (derived quantity)
    #  Minimum Jump Duration s (duplicated for some reason)

    # find: initial velocity, gravity, additive force, max duration (time
    # after which longer button doesn't help)

    for v, jvec in enumerate(jumpInputs):
        print("LOAD " + str(v))
        emu.load(start)
        emu.step(0x0, 0x0)
        total = total + 1
        y_posns = []
        ram_ptr = emu.fc.fceu.RAM
        y_posns.append(ram_ptr[mario_y])
        print("Start y:" + str(y_posns[0]))
        state = J_START
        trise = 0
        drise = 0
        vrise = 0
        tfall = 0
        dfall = 0
        vfall = 0
        for i, m in enumerate(jvec):
            # print "Step: %d, %d" % (i, m)
            total = total + 1
            emu.step(m, 0x0)
            y_posns.append(ram_ptr[mario_y])
            dy = y_posns[-1] - y_posns[-2]
            # print("y:" + str(y_posns[-1]))
            # print("dy:" + str(dy))
            if state == J_RISE:
                trise += 1
                drise += dy
            if state == J_FALL:
                tfall += 1
                dfall += dy
            if state == J_START and y_posns[-1] != y_posns[0]:
                state = J_RISE
                print "UP!!"
            if state == J_RISE and dy >= 0:
                state = J_FALL
                vrise = drise / float(trise)
                print "Rise: {} units in {} s -> {} u/s".format(drise, trise, vrise)
                print "APEX"
            if state == J_FALL and y_posns[-1] == y_posns[0]:
                state = J_END
                vfall = dfall / float(tfall)
                print "Fall: {} units in {} s -> {} u/s".format(dfall, tfall, vfall)
                print "LAND"
                break
    print "Total steps:" + str(total)


if __name__ == "__main__":
    go('mario.nes')
