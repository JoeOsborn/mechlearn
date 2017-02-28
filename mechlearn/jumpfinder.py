import fceulib
from fceulib import VectorBytes
from PIL import Image
import numbers
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Stat:

    def __init__(self, v, smooth):
        self.window = smooth
        self.lastVal = float(v)
        self.observedVals = 1
        self.allVals = []
        if smooth is None:
            self.netVal = v
            self.observedVals = 1.0
        elif smooth == "interesting_mode":
            self.initVal = float(v)
            self.vals = [float(v)]
        else:
            self.vals = [float(v)]
            self.netVal = float(v)
            for i in range(1, smooth):
                self.vals.append(0.0)
            self.idx = 1

    def __repr__(self):
        return "Stat(" + str(self.val()) + "," + str(self.window) + ")"

    def val(self):
        if self.window is None:
            return self.netVal / self.observedVals
        elif self.window == "interesting_mode":
            counts = dict()
            for v in self.vals:
                counts[v] = counts.get(v, 0) + 1
            sortedCounts = counts.items()
            sortedCounts.sort()
            threshold = math.ceil(math.log(self.observedVals))
            if len(sortedCounts) >= 2:
                threshold = max(threshold, sortedCounts[1][1])
            if sortedCounts[0][1] > threshold:
                return sortedCounts[0][0]
            else:
                return self.initVal
        else:
            return self.netVal / float(min(self.window, self.observedVals))

    def update(self, value, clobber=False):
        self.allVals.append(float(value))
        self.lastVal = float(value)
        self.observedVals += 1.0
        if self.window is None:
            if clobber:
                self.netVal = float(value)
                self.observedVals = 1.0
            else:
                self.netVal += float(value)
        elif self.window == "interesting_mode":
            if clobber:
                self.observedVals = 1.0
                self.vals = []
            self.vals.append(float(value))
        else:
            if clobber:
                self.vals[0] = float(value)
                self.netVal = float(value)
                for i in range(1, self.window):
                    self.vals[i] = 0.0
                self.idx = 1
                self.observedVals = 1.0
            else:
                self.netVal -= self.vals[self.idx]
                self.netVal += float(value)
                self.vals[self.idx] = float(value)
                self.idx = (self.idx + 1) % self.window

    def steady(self, frac=0.5):
        # Last value within frac% of avg value
        print "Steady? " + "LV:" + str(self.lastVal) + ", V:" + str(self.val()) + " : " + str(abs(self.lastVal - self.val())) + " vs " + str(abs(self.val() * float(frac)))
        return abs(self.lastVal - self.val()) <= abs(self.val() * float(frac))


class HA:

    def __init__(self, params, vbls, constraints, states, initial):
        self.params = params
        self.variables = {}
        self.variableNames = set()
        for k, v in vbls.items():
            self.variableNames.add(k)
            self.variables[(k, 0)] = float(v)
            self.variables[(k, 1)] = 0.0
            self.variables[(k, 2)] = 0.0
        self.constraints = constraints
        self.states = states
        self.initial = initial

    def makeValuation(self, inits):
        values = self.variables.copy()
        values.update(inits)
        for k, v in values.items():
            values[k] = float(v)
        return HAVal(values, self.initial, 0.0)

    def bound(self, val, var):
        if var in self.constraints:
            cur = val.variables[var]
            lo = self.toValue(self.constraints[var][0], val)
            hi = self.toValue(self.constraints[var][1], val)
            if cur < lo:
                val.variables[var] = lo
            elif cur > hi:
                val.variables[var] = hi

    def step(self, val, dt, buttons, collisions):
        s = self.states[val.state]
        # TODO: continuous before discrete?
        self.discreteStep(val, s.transitions, buttons, collisions)
        self.continuousStep(val, s.flows, dt, collisions)

    def continuousStep(self, val, flows, dt, collisions):
        for v in self.variableNames:
            x = self.toValue(flows.get((v, 0), 0.0), val)
            dx = self.toValue(flows.get((v, 1), 0.0), val)
            ddx = self.toValue(flows.get((v, 2), 0.0), val)
            assert isinstance(x, numbers.Number), str(x)
            assert isinstance(dx, numbers.Number), str(dx)
            assert isinstance(ddx, numbers.Number), str(ddx)
            oldX = val.variables[(v, 0)]
            oldDX = val.variables[(v, 1)]
            if x != 0:
                val.variables[(v, 0)] = x
                val.variables[(v, 1)] = x - oldX
                val.variables[(v, 2)] = val.variables[(v, 1)] - oldDX
            elif dx != 0:
                val.variables[(v, 0)] += dx * dt
                self.bound(val, (v, 0))
                val.variables[(v, 1)] = dx
                val.variables[(v, 2)] = val.variables[(v, 1)] - oldDX
            elif ddx != 0:
                val.variables[(v, 2)] = ddx
                val.variables[(v, 1)] += ddx * dt
                self.bound(val, (v, 1))
                val.variables[(v, 0)] += val.variables[(v, 1)] * dt
                self.bound(val, (v, 0))
        val.timeInState += dt
        val.time += dt

    def discreteStep(self, val, transitions, buttons, collisions):
        for t in transitions:
            if t.guardSatisfied(self, val, buttons, collisions):
                # print ("Follow: " + val.state + " -> " + t.target + " via " +
                #        str(t.guard))
                new_vars = dict()
                for k, v in (t.update or {}).items():
                    new_vars[k] = self.toValue(v, val)
                    assert isinstance(new_vars[k], numbers.Number), str(new_vars[k])
                for k, v in new_vars.items():
                    val.variables[k] = v
                if val.state != t.target:
                    val.state = t.target
                    val.timeInState = 0
                break

    def toValue(self, expr, valuation):
        if isinstance(expr, Stat):
            return expr.val()
        elif expr in self.params:
            p = self.params[expr]
            return self.toValue(p, valuation)
        elif expr in valuation.variables:
            return float(valuation.variables[expr])
        elif isinstance(expr, tuple):
            if expr[0] == "max":
                assert len(expr) == 3
                return max(self.toValue(expr[1], valuation),
                           self.toValue(expr[2], valuation))
            elif expr[0] == "min":
                assert len(expr) == 3
                return min(self.toValue(expr[1], valuation),
                           self.toValue(expr[2], valuation))
            elif expr[0] == "clip":
                assert len(expr) == 4
                v = self.toValue(expr[1], valuation)
                return min(max(v,
                               self.toValue(expr[2], valuation)),
                           self.toValue(expr[3], valuation))
            elif expr[0] == "abs":
                return abs(self.toValue(expr[1], valuation))
            elif expr[0] == "+":
                assert len(expr) == 3
                return self.toValue(expr[1], valuation) + self.toValue(expr[2], valuation)
            elif expr[0] == "-":
                assert len(expr) == 3
                return self.toValue(expr[1], valuation) - self.toValue(expr[2], valuation)
            elif expr[0] == "*":
                assert len(expr) == 3
                return self.toValue(expr[1], valuation) * self.toValue(expr[2], valuation)
            elif expr[0] == "/":
                assert len(expr) == 3
                return self.toValue(expr[1], valuation) / self.toValue(expr[2], valuation)
            else:
                raise Exception("Unrecognized expr", expr)
        elif isinstance(expr, numbers.Number):
            return expr
        else:
            raise Exception("Unrecognized expr", expr)


class HAVal:

    def __init__(self, vbls, initial, t, timeInState=0):
        self.variables = vbls
        self.state = initial
        self.time = t
        self.timeInState = timeInState

    def __str__(self):
        return ("HAVal. " + self.state +
                "(" + str(self.timeInState) + ")" +
                "\n  " + str(self.variables) +
                "\n  Alive: " + str(self.time))

    def copy(self):
        return HAVal(self.variables.copy(),
                     self.state,
                     self.time,
                     self.timeInState)


class HAState:

    def __init__(self, flows, transitions):
        self.flows = flows
        self.transitions = transitions


class HATransition:

    def __init__(self, target, guard, update):
        self.target = target
        self.guard = guard
        self.update = update

    def guardSatisfied(self, m, val, buttons, collisions):
        for g in self.guard:
            if not self.primitiveGuardSatisfied(m,
                                                g,
                                                val,
                                                buttons,
                                                collisions):
                return False
        return True

    def primitiveGuardSatisfied(self, m, g, val, buttons, collisions):
        gt = g[0]
        if gt == "not":
            return not self.primitiveGuardSatisfied(m, g[1], val, buttons, collisions)
        elif gt == "button":
            # TODO: distinguish pressed/on and released/off
            if g[1] == "pressed" and g[2] in buttons:
                return True
            if g[1] == "on" and g[2] in buttons:
                return True
            if g[1] == "released" and g[2] not in buttons:
                return True
            if g[1] == "off" and g[2] not in buttons:
                return True
            return False
        elif gt == "colliding":
            for side, ctype in collisions:
                if g[1] == side and g[2] == ctype:
                    return True
            return False
        elif gt == "timer":
            return val.timeInState >= m.toValue(g[1], val)
        elif gt == "gt":
            return m.toValue(g[1], val) > m.toValue(g[2], val)
        elif gt == "gte":
            return m.toValue(g[1], val) >= m.toValue(g[2], val)
        elif gt == "eq":
            return m.toValue(g[1], val) == m.toValue(g[2], val)
        elif gt == "lte":
            return m.toValue(g[1], val) <= m.toValue(g[2], val)
        elif gt == "lt":
            return m.toValue(g[1], val) < m.toValue(g[2], val)
        else:
            raise Exception("Unrecognized Guard", g)

DT = 1.0 / 60.0

"""
There are three hypotheses we're interested in.

1. First, does this jump vary depending on duration of holding the button?
2. Second, is there acceleration while ascending,
   or is Y velocity fixed for some duration?
3. Finally, does hitting the ceiling kill y-velocity?

As it turns out, 1 and 2 are captured by the mario model as-is
(1 may turn into a 1-frame "maxButtonDuration" and
 2 may result in "risingGravity = 0" and
   earlyOutClipVel = Infinity or jump start vel),
and we can start accounting for 3 later (maybe with a Boolean parameter?).
"""

marioModel = HA(
    {"gravity": Stat(0.0, None),
     "groundToUpControlDYReset": Stat(0.0, None),
     "upFixedToDownDYReset": Stat(0.0, None),
     "upControlToDownDYReset": Stat(0.0, None),
     "downToGroundDYReset": Stat(0.0, None),
     "terminalVY": Stat(100000.0, None),
     "up-control-gravity": Stat(0.0, None),
     "up-fixed-gravity": Stat(0.0, None),
     "maxButtonDuration": Stat(100000.0, None),
     "minButtonDuration": Stat(0.0, None),
     "upControlToUpFixedDYReset": Stat(0.0, None)},
    {"x": 0, "y": 0},
    {("y", 1): (-1000000.0, "terminalVY")},
    {
        "ground": HAState({("y", 1): 0}, [
            HATransition("up-control",
                         [("button", "pressed", "jump")],
                         {("y", 1): "groundToUpControlDYReset"}),
            HATransition("down",
                         [("not", ("colliding", "bottom", "ground"))],
                         None)
        ]),
        "up-control": HAState({("y", 2): "up-control-gravity"}, [
            # This edge may not always be present.
            HATransition("down",
                         [("colliding", "top", "ground")],
                         {("y", 1): "upControlToDownDYReset"}),
            HATransition("down",
                         [("gte", ("y", 1), 0)],
                         {("y", 1): "upControlToDownDYReset"}),
            HATransition("up-fixed",
                         [("timer", "maxButtonDuration")],
                         {("y", 1): "upControlToUpFixedDYReset"}),
            HATransition("up-fixed",
                         [("button", "off", "jump"),
                          ("timer", "minButtonDuration")],
                         {("y", 1): "upControlToUpFixedDYReset"})
        ]),
        "up-fixed": HAState({("y", 2): "up-fixed-gravity"}, [
            # This edge may not always be present.
            HATransition("down",
                         [("colliding", "top", "ground")],
                         {("y", 1): "upFixedToDownDYReset"}),
            HATransition("down",
                         [("gte", ("y", 1), 0)],
                         {("y", 1): "upFixedToDownDYReset"})
        ]),
        "down": HAState({("y", 2): "gravity"}, [
            # this edge may not always be present.
            # It's here too for the case where we are in down state
            # but our velocity is still positive.
            HATransition("down",
                         [("colliding", "top", "ground")],
                         None),
            HATransition("ground",
                         [("colliding", "bottom", "ground")],
                         {("y", 1): "downToGroundDYReset"})
        ]),
        "dead": HAState({("y", 1): 0}, [])
    },
    "ground"
)


class Stats:

    def __init__(self, x, y, smooth=5):
        self.x = Stat(x, smooth)
        self.y = Stat(y, smooth)
        self.dx = Stat(0, smooth)
        self.dy = Stat(0, smooth)
        self.ddx = Stat(0, smooth)
        self.ddy = Stat(0, smooth)
        self.smooth = smooth
        self.firstUpdate = True

    def update(self, x, y):
        lx = self.x.lastVal
        self.x.update(x)
        ly = self.y.lastVal
        self.y.update(y)
        ldx = self.dx.val()
        dx = x - lx
        self.dx.update(dx / DT, self.firstUpdate)
        ldy = self.dy.val()
        dy = y - ly
        self.dy.update(dy / DT, self.firstUpdate)
        ddx = self.dx.val() - ldx
        self.ddx.update(ddx / DT, self.firstUpdate)
        ddy = self.dy.val() - ldy
        self.ddy.update(ddy / DT, self.firstUpdate)
        self.firstUpdate = False


def calcErrorStep(model, val,
                  emulator, xget, yget,
                  jumpButton,
                  m,
                  stats):
    model.step(val, DT, set(["jump"] if m & jumpButton else []), set([]))
    emulator.step(m, 0x0)
    nowX = xget(emulator)
    nowY = yget(emulator)
    stats.update(nowX, nowY)
    hereErrors = (stats.x.val() - val.variables[("x", 0)],
                  stats.y.val() - val.variables[("y", 0)],
                  stats.dx.val() - val.variables[("x", 1)],
                  stats.dy.val() - val.variables[("y", 1)],
                  stats.ddx.val() - val.variables[("x", 2)],
                  stats.ddy.val() - val.variables[("y", 2)])
    return hereErrors


def calcError(model, emulator, xget, yget, jumpButton, actions):
    """
    Run the model and the game through the same steps.
    Accumulate and return absolute errors in x, y, dx, dy, ddx, ddy,
    both net and in each discrete state of the model.
    """
    startX = xget(emulator)
    startY = yget(emulator)
    v = model.makeValuation({("x", 0): startX, ("y", 0): startY})
    print "Start state:" + v.state
    errorsByFrame = []
    netErrorsByState = {}
    for k in model.states:
        netErrorsByState[k] = [0, 0, 0, 0, 0, 0]
    netErrors = [0, 0, 0, 0, 0, 0]
    grossErrors = [0, 0, 0, 0, 0, 0]
    stats = Stats(startX, startY)
    estStats = Stats(startX, startY)
    for (i, m) in enumerate(actions):
        # TODO: buttons for real
        # TODO: collisions
        hereErrors = calcErrorStep(
            model, v, emulator,
            xget, yget, jumpButton,
            m,
            stats
        )
        estStats.y.update(v.variables[("y", 0)])
        estStats.dy.update(v.variables[("y", 1)])
        # TODO: hack because I don't deal with collision yet for the model
        if yget(emulator) == startY and i > 5:
            break
        print "F:" + str(i) + " J?:" + str(True if m & jumpButton else False)
        print str(v)
        print "Y:" + str(yget(emulator)) + " DY:" + str(stats.dy.val()) + " DDY:" + str(stats.ddy.val())
        print "here errors:" + str(hereErrors)
        for i in range(0, 6):
            netErrors[i] += hereErrors[i]
            grossErrors[i] += abs(hereErrors[i])
            netErrorsByState[v.state][i] += abs(hereErrors[i])
        errorsByFrame.append(hereErrors)
    plt.figure(1)
    plt.plot(stats.y.allVals, '+-')
    plt.plot(estStats.y.allVals, "x-")
    plt.gca().invert_yaxis()
    plt.savefig('test-ys')
    plt.close(1)
    plt.figure(2)
    plt.plot(stats.dy.allVals, '+-')
    plt.plot(estStats.dy.allVals, "x-")
    plt.gca().invert_yaxis()
    plt.savefig('test-dys')
    plt.close(2)
    return (netErrors, grossErrors, errorsByFrame, netErrorsByState)


RIGHT = 1 << 7
LEFT = 1 << 6
DOWN = 1 << 5
UP = 1 << 4
START = 1 << 3
SELECT = 1 << 2
B = 1 << 1
A = 1 << 0

mario_x = 0x006D
mario_y = 0x00CE

metroid_x = 0x0051
metroid_y = 0x0052

cv_x = 0x038C
cv_y = 0x0354

mm_x = 0x022
mm_y = 0x025


imgBuffer = VectorBytes()


def hold(mask, duration):
    return [mask for i in range(duration)]


def outputImage(emu, name, buf=imgBuffer):
    emu.imageInto(buf)
    outImg = Image.frombytes("RGBA", (256, 256), str(bytearray(buf)))
    outImg.save(name + ".png")


def marioGetX(emu):
    return emu.fc.fceu.RAM[mario_x]


def marioGetY(emu):
    return emu.fc.fceu.RAM[mario_y]


def cvGetX(emu):
    return emu.fc.fceu.RAM[cv_x]


def cvGetY(emu):
    return emu.fc.fceu.RAM[cv_y]


def mmGetX(emu):
    return emu.fc.fceu.RAM[mm_x]


def mmGetY(emu):
    return emu.fc.fceu.RAM[mm_y]


def metroidGetX(emu):
    return emu.fc.fceu.RAM[metroid_x]


def metroidGetY(emu):
    return emu.fc.fceu.RAM[metroid_y]


def findJumpHoldLimit(emu, start, getx, gety, jumpButton, model):
    # FIXME: Quite domain specific
    maxHeldFrames = 120
    jumpInputs = [hold(jumpButton, t) + hold(0x0, 600)
                  for t in range(1, maxHeldFrames + 1)]
    longestJump = 0
    veryShortestJump = 0
    for (i, jvec) in enumerate(jumpInputs):
        startX = getx(emu)
        startY = gety(emu)
        stats = Stats(startX, startY, 5)
        val = model.makeValuation({("x", 0): startX,
                                   ("y", 0): startY})
        for (j, move) in enumerate(jvec):
            calcErrorStep(
                model, val, emu,
                getx, gety, jumpButton,
                move,
                stats
            )
            if stats.y.lastVal == startY and j > 5:
                thisJumpDuration = (j + 1) * DT
                thisButtonHold = (i + 1) * DT
                if i == 0:
                    veryShortestJump = thisJumpDuration
                if abs(thisJumpDuration - veryShortestJump) < DT:
                    model.params["minButtonDuration"].update(
                        thisButtonHold,
                        True
                    )
                # print "Jump duration " + str(thisJumpDuration)
                if thisJumpDuration > longestJump:
                    longestJump = thisJumpDuration
                    model.params["maxButtonDuration"].update(
                        # The button is pushed at the beginning of this frame,
                        # but the timer doesn't start ticking until the next.
                        thisButtonHold - DT,
                        True
                    )
                break
        emu.load(start)
    print "Jump hold min: " + str(model.params["minButtonDuration"].val()) + " Shortest jump: " + str(veryShortestJump)
    print "Jump hold max: " + str(model.params["maxButtonDuration"].val()) + " Longest jump: " + str(longestJump)
    if longestJump > model.params["longestJump"].val():
        model.params["longestJump"].update(longestJump, True)
    return model.params["maxButtonDuration"].val()


def findJumpAccs(emu, start, getx, gety, jumpButton, model):
    minHeldFrames = int(
        math.floor(model.params["minButtonDuration"].val() / DT))
    maxHeldFrames = int(
        math.ceil(model.params["maxButtonDuration"].val() / DT))
    maxWaitFrames = int(math.ceil(model.params["longestJump"].val() / DT))
    jumpInputs = [hold(jumpButton, t) + hold(0x0, maxWaitFrames)
                  for t in range(minHeldFrames, maxHeldFrames + 1)]

    for (i, jvec) in enumerate(jumpInputs):
        startX = getx(emu)
        startY = gety(emu)
        stats = Stats(startX, startY, 5)
        startedJump = False
        val = model.makeValuation({("x", 0): startX,
                                   ("y", 0): startY})
        jumpStartDY = 0
        jumpFallStartFrame = 0
        print "Next"
        for (j, move) in enumerate(jvec):
            lastDY = stats.dy.val()
            calcErrorStep(
                model, val, emu,
                getx, gety, jumpButton,
                move,
                stats
            )
            # FIXME: The below conditions are all quite hacky
            # too fine-tuned for Mario, Metroid jump does not start until
            # subsequent frame!  Should check for "velocity increased from
            # initial 0"
            if j == stats.smooth:  # our first transition
                # eeeeeh, close but not quite there. some of this starting vel
                # is still counting as an upward acceleration during the
                # upwards jump part, which seems wrong? Do we speed up while
                # the button is being held? Not 100% sure.  is metroid behavior
                # documented??
                # print "JSS:" + str(j) + ":" + str(stats.dy.val())
                startedJump = True
                model.params["jumpStartSpeed"].update(
                    stats.dy.val(), i == 0)
                jumpStartDY = stats.dy.val()
                # print "-->" + str(model.params["jumpStartSpeed"].val())
            # emu state == UP, perfect knowledge
            elif startedJump and (jvec[j] & jumpButton):
                pass
                # print "RG:" + str(stats.ddy.lastVal)
                # model.params["risingGravity"].update(
                #    stats.ddy.lastVal, i == 0 and j == stats.smooth + 1)
                # print "-->" + str(model.params["risingGravity"].val())
            elif startedJump and stats.y.lastVal != startY:  # emu state == DOWN, perfect knowledge
                # print "G:" + str(stats.ddy.lastVal)
                # learn earlyOutClipVel if jvec[j-1] & jumpButton, but
                # don't try to learn gravity on that frame!
                if jvec[j - 1] & jumpButton:
                    netDY = stats.dy.val() - jumpStartDY
                    print "Time to fall-start:" + str(j * DT)
                    print "Net DY:" + str(netDY) + "; GuessedRGrav:" + str(netDY / (j * DT))
                    model.params["risingGravity"].update(
                        netDY / (j * DT), i == 0)
                    jumpFallStartFrame = j
                    # just transitioned to falling.
                    # We know that because by construction we don't jump (much)
                    # longer than it helps to do so.
                    acc = model.params["risingGravity"].val()
                    expectedDY = lastDY + acc * DT
                    difference = stats.dy.lastVal - expectedDY
                    print "DY: " + str(stats.dy.lastVal) + " Expected: " + str(expectedDY) + " delta: " + str(difference)
                    # TODO: Should only update earlyOutClipVel if I _usually_
                    # reset to the same value.
                    if (difference >
                            model.params["risingGravity"].val() * DT * 2 or
                        difference >
                            model.params["gravity"].val() * DT * 2):
                        print "Clip Y vel from:" + str(lastDY) + " to " + str(stats.dy.val()) + "/" + str(stats.dy.lastVal)
                        model.params["earlyOutClipVel"].update(
                            stats.dy.lastVal,
                            # HACK: to clobber the giant initial value
                            len(model.params[
                                "earlyOutClipVel"].allVals) == 1
                        )
                else:
                    pass
                    # aggregate falling gravity
                    # model.params["gravity"].update(
                    #    stats.ddy.lastVal, i == 0 and (jvec[j - 1] & jumpButton))
                    # print "-->" + str(model.params["gravity"].val())
                    # jump hasn't started yet? probably won't ever!
            elif (not startedJump and j > stats.smooth):
                break
            elif (startedJump and j > stats.smooth and stats.y.lastVal == startY):
                netDY = stats.dy.val()
                f = j - jumpFallStartFrame
                print "Time to ground:" + str(f * DT)
                print "Net DY:" + str(netDY) + "; GuessedGGrav:" + str(netDY / (f * DT))
                model.params["gravity"].update(netDY / (f * DT), i == 0)
                break
        plt.figure(1)
        plt.plot(stats.y.allVals, '+-')
        plt.figure(2)
        plt.gca().invert_yaxis()
        plt.plot(stats.dy.allVals, 'x-')
        plt.savefig('dys' + str(i))
        plt.close(2)
        emu.load(start)
    plt.figure(1)
    plt.title('Yvals')
    plt.gca().invert_yaxis()
    plt.savefig('ys')
    plt.close(1)
    print "Param Results:"
    print "JSS>" + str(model.params["jumpStartSpeed"].val())
    print "RG->" + str(model.params["risingGravity"].val())
    print "CLP>" + str(model.params["earlyOutClipVel"].val())
    print "G-->" + str(model.params["gravity"].val())


def runTrials(emu, start, getx, gety, jumpButton):
    # FIXME: Quite domain specific
    maxHeldFrames = 120
    minHeldFrames = 1
    jumpInputs = [hold(jumpButton, t) + hold(0x0, 600)
                  for t in range(minHeldFrames, maxHeldFrames + 1)]
    startX = getx(emu)
    startY = gety(emu)
    trials = []
    for (i, jvec) in enumerate(jumpInputs):
        stats = Stats(startX, startY, 5)
        for (j, move) in enumerate(jvec):
            emu.step(move, 0x0)
            nowX = getx(emu)
            nowY = gety(emu)
            #print j, move, nowX, nowY
            # can also aggregate other useful stuff into stats later
            stats.update(nowX, nowY)
            if gety(emu) == startY and j > 5:
                break
        trials.append((jvec[0:j], stats))
        plt.figure(1)
        plt.plot(stats.y.allVals, '+-')
        plt.figure(2)
        plt.gca().invert_yaxis()
        plt.plot(stats.dy.allVals, 'x-')
        plt.savefig('trials/dys' + str(i))
        plt.close(2)
        emu.load(start)
    # find first index when count of moves does not go up
    shortest = len(trials[0][0])
    longest = len(trials[-1][0])
    shortesti = minHeldFrames-1
    longesti = maxHeldFrames-1
    for i, (moves, stats) in enumerate(trials):
        if len(moves) == shortest:
            shortesti = max(i,shortesti)
        if len(moves) == longest:
            longesti = min(i,longesti)
    maxHeldFrames = longesti + 1
    minHeldFrames = min(maxHeldFrames, shortesti + 1)
    plt.figure(1)
    plt.title('Yvals')
    plt.gca().invert_yaxis()
    plt.savefig('trials/ys')
    plt.close(1)
    return trials[minHeldFrames-1:maxHeldFrames], minHeldFrames, maxHeldFrames

# Moving trials:
# start moving right, find the fastest speed
# then try trials at min speed, max * 0.25, max * 0.5, max * 0.75, max with shortest hold and longest hold.

def go():
    jumpButton = A
    games = {
        "mario": (
            "mario.nes",
            (hold(0x0, 120) + hold(START | jumpButton, 30) +
             hold(0x0, 150)),
            marioGetX,
            marioGetY
        ),
        "metroid": (
            "metroid.nes",
            (hold(0x0, 60) + hold(START, 1) + hold(0x0, 15) +
             hold(START, 1) + hold(0x0, 600) + hold(LEFT, 400) +
             hold(RIGHT, 30)),
            metroidGetX,
            metroidGetY
        )
    }
    (game, startInputs, getx, gety) = games["mario"]
    total = 0
    emu = fceulib.runGame(game)
    for m in startInputs:
        total = total + 1
        emu.step(m, 0x0)
    total = total + 1
    emu.step(0x0, 0x0)
    start = VectorBytes()
    print("SAVE")
    emu.save(start)
    print("SAVED")
    outputImage(emu, "start")
    findJumpHoldLimit(emu, start, getx, gety, jumpButton, marioModel)
    findJumpAccs(emu, start, getx, gety, jumpButton, marioModel)

    maxJump = marioModel.params["maxButtonDuration"].val()
    testInputs = (hold(jumpButton, int(maxJump / DT) + 1) +
                  hold(0x0, int(0.8 / DT)))
    emu.load(start)
    # TODO: see if error is improved by using either guessedWhatever or the frame by frame Whatever (is this the same as closed vs open loop?)
    # TODO: Analyze the errors.  First off, it looks like model.y is wrong from "down" onwards.  It also never quite gets as high as the real emulator does in Y.
    #  It looks like there's also weirdness (expected weirdness?) around landing. I think if I told the model when y=startY as a collision, we would be well good. A position error of 51 seems pretty wrong
    #  ... rising accel is not too far off (compared to an estimate of r=g=300 it's still too strong I think?). falling accel is way overestimated (and in fact is not seemingly that great for down either....?). why?
    (net, gross, byStep, byState) = calcError(marioModel,
                                              emu, getx, gety,
                                              jumpButton, testInputs)
    print "Gross error:" + str(gross)
    print "Net error:" + str(net)
    print "By state:" + str(byState)
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
    #  Additive Jump Force, Threshold u/s2
    #  Release Drag u/s2
    #  Hold Jump Input (this refers to holding horizontal while jumping)
    #  Minimum Jump Height u (derived quantity)
    #  Minimum Jump Duration s (duplicated for some reason)

if __name__ == "__main__":
    go()
