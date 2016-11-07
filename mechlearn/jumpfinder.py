import fceulib
from fceulib import VectorBytes
from PIL import Image
import numbers
import math

"""
Character learning has two components.  First, we need to learn the structure of the automaton representing the automaton.  ProbFOIL, a relational learning library built on top of ProbLog, seems a natural choice---at any rate, it remains for future work.

Once we have a structure, we can fit parameters to that structure.  Example parameters include the dynamics of each state and any parameters on guards.  Since we only consider one character at a time, we can abstract the environment into occluding collisions, non-occluding collisions, and hostile collisions.

We'll work with non-hierarchical, non-concurrent hybrid automata for now.  Also, we'll assume left and right movement are mirrored.  Our representation of a state will define, for each variable, whether it has a constant velocity or a constant acceleration (and what that value is); and what transitions are available under what circumstances (a conjunction of abstracted inputs, continuous variable thresholds, timers, any collisions of any type at any normal to the character, and negations of the same).  A state machine is a set of states and an initial state.
"""


class Stat:

    def __init__(self, v, smooth):
        self.window = smooth
        self.lastVal = float(v)
        self.observedVals = 1
        if smooth is None:
            self.netVal = v
            self.observedVals = 1.0
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
        else:
            return self.netVal / float(min(self.window, self.observedVals))

    def update(self, value, clobber=False):
        self.lastVal = float(value)
        self.observedVals += 1.0
        if self.window is None:
            if clobber:
                self.netVal = float(value)
                self.observedVals = 1.0
            else:
                self.netVal += float(value)
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

    def step(self, val, dt, buttons):
        s = self.states[val.state]
        self.continuousStep(val, s.flows, dt)
        self.discreteStep(val, s.transitions, buttons)

    def continuousStep(self, val, flows, dt):
        for v in self.variableNames:
            x = self.toValue(flows.get((v, 0), 0.0), val)
            dx = self.toValue(flows.get((v, 1), 0.0), val)
            ddx = self.toValue(flows.get((v, 2), 0.0), val)
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

    def discreteStep(self, val, transitions, buttons):
        for t in transitions:
            if t.guardSatisfied(self, val, buttons):
                # print ("Follow: " + val.state + " -> " + t.target + " via " +
                #        str(t.guard))
                for k, v in (t.update or {}).items():
                    val.variables[k] = self.toValue(v, val)
                if val.state != t.target:
                    val.state = t.target
                    val.timeInState = 0
                break

    def toValue(self, expr, valuation):
        if expr in self.params:
            p = self.params[expr]
            if isinstance(p, Stat):
                return p.val()
            else:
                return float(p)
        elif expr in valuation.variables:
            return float(valuation.variables[expr])
        elif isinstance(expr, tuple) and expr[0] == "max":
            return max(self.toValue(expr[1], valuation),
                       self.toValue(expr[2], valuation))
        elif isinstance(expr, tuple) and expr[0] == "min":
            return min(self.toValue(expr[1], valuation),
                       self.toValue(expr[2], valuation))
        elif isinstance(expr, numbers.Number):
            return expr
        print "Default expr:" + str(expr) + " type:" + str(type(expr))
        return expr


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

    def guardSatisfied(self, m, val, buttons):
        for g in self.guard:
            return self.primitiveGuardSatisfied(m, g, val, buttons)

    def primitiveGuardSatisfied(self, m, g, val, buttons):
        gt = g[0]
        if gt == "not":
            return not self.primitiveGuardSatisfied(m, g[1], val, buttons)
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
            # TODO
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
            raise ("Unrecognized Guard", g)

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
    {"gravity": Stat(0, None),
     "jumpStartSpeed": Stat(0, None),
     "terminalVY": Stat(0, None),
     "risingGravity": Stat(0, None),
     "maxButtonDuration": Stat(0, None),
     "earlyOutClipVel": Stat(0, None),
     # Just for curiosity/info
     "longestJump": Stat(0, None)},
    {"x": 0, "y": 0},
    {("y", 1): (-200, "terminalVY")},
    {
        "ground": HAState({("y", 1): 0}, [
            HATransition("up",
                         [("button", "pressed", "jump")],
                         {("y", 1): "jumpStartSpeed"}),
            HATransition("down",
                         [("not", ("colliding", "bottom", "ground"))],
                         None)
        ]),
        "up": HAState({("y", 2): "risingGravity"}, [
            # This edge may not always be present
            HATransition("down",
                         [("colliding", "top", "ground")],
                         {("y", 1): 0}),
            HATransition("down",
                         [("timer", "maxButtonDuration")],
                         None),
            HATransition("down",
                         [("button", "off", "jump")],
                         {("y", 1): ("min", ("y", 1), "earlyOutClipVel")})
        ]),
        "down": HAState({("y", 2): "gravity"}, [
            # not always present, here for the case where we are in down state
            # but our velocity is still positive
            HATransition("down",
                         [("colliding", "top", "ground")],
                         {("y", 1): 0}),
            HATransition("ground",
                         [("colliding", "bottom", "ground")],
                         None)
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
    model.step(val, DT, set(["jump"] if m & jumpButton else []))
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
    print "Start state:"+v.state
    errorsByFrame = []
    netErrorsByState = {}
    for k in model.states:
        netErrorsByState[k] = [0, 0, 0, 0, 0, 0]
    netErrors = [0, 0, 0, 0, 0, 0]
    stats = Stats(startX, startY)
    for (i,m) in enumerate(actions):
        # TODO: buttons for real
        # TODO: collisions
        hereErrors = calcErrorStep(
            model, v, emulator,
            xget, yget, jumpButton,
            m,
            stats
        )
        print "F:"+str(i)+" J?:"+str(True if m&jumpButton else False)
        print "MState:"+str(v.state)
        print "Y:"+str(yget(emulator))
        print "MY:"+str(v.variables[("y",0)])
        print "here errors:"+str(hereErrors)
        for i in range(0, 6):
            netErrors[i] += abs(hereErrors[i])
            netErrorsByState[v.state][i] += abs(hereErrors[i])
        errorsByFrame.append(hereErrors)
    return (netErrors, errorsByFrame, netErrorsByState)


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

def metroidGetX(emu):
    return emu.fc.fceu.RAM[metroid_x]

def metroidGetY(emu):
    return emu.fc.fceu.RAM[metroid_y]


def findJumpHoldLimit(emu, start, getx, gety, jumpButton, model):
    maxHeldFrames = 120
    jumpInputs = [hold(jumpButton, t) + hold(0x0, 600)
                  for t in range(1, maxHeldFrames + 1)]
    longestJump = 0
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
                # print "Jump duration " + str(thisJumpDuration)
                if thisJumpDuration > longestJump:
                    longestJump = thisJumpDuration
                    thisButtonHold = (i + 1) * DT
                    if thisButtonHold > model.params["maxButtonDuration"].val():
                        # print "Longer limit: " + str(model.params["maxButtonDuration"].val()) + " -> " + str(thisButtonHold - DT)
                        # The button is pushed at the beginning of this frame,
                        # but the timer doesn't start ticking until the next.
                        model.params["maxButtonDuration"].update(
                            thisButtonHold - DT,
                            True
                        )
                break
        emu.load(start)
    print "Jump hold limit: " + str(model.params["maxButtonDuration"].val()) + " Longest jump: " + str(longestJump)
    if longestJump > model.params["longestJump"].val():
        model.params["longestJump"].update(longestJump, True)
    return model.params["maxButtonDuration"].val()


def findJumpAccs(emu, start, getx, gety, jumpButton, model):
    maxHeldFrames = int(
        math.ceil(model.params["maxButtonDuration"].val() / DT))
    maxWaitFrames = int(math.ceil(model.params["longestJump"].val() / DT))
    jumpInputs = [hold(jumpButton, t) + hold(0x0, maxWaitFrames)
                  for t in range(1, maxHeldFrames + 1)]

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
            # too fine-tuned for Mario, Metroid jump does not start until subsequent frame!  Should check for "velocity increased from initial 0"
            print "DY:"+str(stats.dy.lastVal)
            if j == stats.smooth:  # our first transition
                ###eeeeeh, close but not quite there. some of this starting vel is still counting as an upward acceleration during the upwards jump part, which seems wrong? Do we speed up while the button is being held? Not 100% sure.  is metroid behavior documented??
                print "JSS:" + str(j)+":"+str(stats.dy.val())
                startedJump = True
                model.params["jumpStartSpeed"].update(
                    stats.dy.val(), i == 0)
                jumpStartDY = stats.dy.val()
                # print "-->" + str(model.params["jumpStartSpeed"].val())
            elif startedJump and (jvec[j] & jumpButton):  # emu state == UP, perfect knowledge
                print "RG:" + str(stats.ddy.lastVal)
                model.params["risingGravity"].update(
                    stats.ddy.lastVal, i == 0 and j == 1)
                print "-->" + str(model.params["risingGravity"].val())
            elif startedJump and stats.y.lastVal != startY:  # emu state == DOWN, perfect knowledge
                print "G:" + str(stats.ddy.lastVal)
                # learn earlyOutClipVel if jvec[j-1] & jumpButton, but
                # don't try to learn gravity on that frame!
                if jvec[j - 1] & jumpButton:
                    netDY = stats.dy.val() - jumpStartDY
                    print "Time to fall-start:"+str(j*DT)
                    print "Net DY:"+str(netDY)+"; GuessedRGrav:"+str(netDY/(j*DT))
                    jumpFallStartFrame = j
                    # just transitioned to falling.
                    # We know that because by construction we don't jump (much)
                    # longer than it helps to do so.
                    acc = model.params["risingGravity"].val()
                    expectedDY = lastDY + acc * DT
                    difference = abs(stats.dy.lastVal - expectedDY)
                    print "DY: " + str(stats.dy.lastVal) + " Expected: " + str(expectedDY) + " delta: " + str(difference)
                    if abs(difference) > 0.1:
                        print "Velocity jump by " + str(difference) + "; Is it constant or an expr?"
                else:
                    # aggregate falling gravity
                    model.params["gravity"].update(
                        stats.ddy.lastVal, i == 0 and (jvec[j - 1] & jumpButton))
                    print "-->" + str(model.params["gravity"].val())
            elif (not startedJump and j > stats.smooth): # jump hasn't started yet? probably won't ever!
                break
            elif (startedJump and j > stats.smooth and stats.y.lastVal == startY):
                netDY = stats.dy.val()
                f = j - jumpFallStartFrame
                print "Time to ground:"+str(f*DT)
                print "Net DY:"+str(netDY)+"; GuessedGGrav:"+str(netDY/(f*DT))
                break
        emu.load(start)
    print "Param Results:"
    print "JSS>" + str(model.params["jumpStartSpeed"].val())
    print "RG->" + str(model.params["risingGravity"].val())
    print "G-->" + str(model.params["gravity"].val())


def go(game):
    total = 0
    emu = fceulib.runGame(game)
    jumpButton = A
    startInputsMario = hold(0x0, 120) + hold(START | jumpButton, 30) + hold(0x0, 150)
    startInputsMetroid = hold(0x0, 60) + hold(START, 1) + hold(0x0, 15) + hold(START, 1) + hold(0x0, 600) + hold(LEFT, 400) + hold(RIGHT, 30)
    startInputs = startInputsMetroid
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
    findJumpHoldLimit(emu, start, metroidGetX, metroidGetY, jumpButton, marioModel)
    findJumpAccs(emu, start, metroidGetX, metroidGetY, jumpButton, marioModel)

    testInputs = hold(jumpButton, int(marioModel.params["maxButtonDuration"].val()/DT)+1) + hold(0x0, int(0.8/DT))
    emu.load(start)
    # TODO: see if error is improved by using either guessedWhatever or the frame by frame Whatever (is this the same as open vs closed loop?)
    # TODO: Analyze the errors.  First off, it looks like model.y is wrong from "down" onwards.  It also never quite gets as high as the real emulator does in Y.
    #  It looks like there's also weirdness (expected weirdness?) around landing. I think if I told the model when y=startY as a collision, we would be well good. A position error of 51 seems pretty wrong.
    (net,_,byState) = calcError(marioModel, emu, metroidGetX, metroidGetY, jumpButton, testInputs)
    print "Net error:"+str(net)
    print "By state:"+str(byState)
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
    go('metroid.nes')
