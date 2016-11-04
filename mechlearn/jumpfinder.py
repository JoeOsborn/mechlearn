import fceulib
from fceulib import VectorBytes
from PIL import Image
import numbers

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
     "earlyOutClipVel": Stat(0, None)},
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
        self.dx.update(dx, self.firstUpdate)
        ldy = self.dy.val()
        dy = y - ly
        self.dy.update(dy, self.firstUpdate)
        ddx = self.dx.val() - ldx
        self.ddx.update(ddx, self.firstUpdate)
        ddy = self.dy.val() - ldy
        self.ddy.update(ddy, self.firstUpdate)
        self.firstUpdate = False


def calcErrorStep(model, val,
                  emulator, xget, yget,
                  m,
                  stats):
    model.step(val, 1.0 / 30.0, set(["jump"] if m & JUMP else []))
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


def calcError(model, emulator, xget, yget, actions):
    """
    Run the model and the game through the same steps.
    Accumulate and return absolute errors in x, y, dx, dy, ddx, ddy,
    both net and in each discrete state of the model.
    """
    startX = xget(emulator)
    startY = yget(emulator)
    v = model.makeValuation({("x", 0): startX, ("y", 0): startY})
    errorsByFrame = []
    netErrorsByState = {}
    for k in model.states:
        netErrorsByState[k] = [0, 0, 0, 0, 0, 0]
    netErrors = [0, 0, 0, 0, 0, 0]
    stats = Stats(startX, startY)
    for m in actions:
        # TODO: buttons for real
        # TODO: collisions
        hereErrors = calcErrorStep(
            model, v, emulator,
            xget, yget,
            m,
            stats
        )
        for i in range(0, 6):
            netErrors[i] += abs(hereErrors[i])
            netErrorsByState[v.state][i] += abs(hereErrors[i])
        errorsByFrame.append(hereErrors)
    return (netErrors, errorsByFrame, netErrorsByState)


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


def marioGetX(emu):
    return emu.fc.fceu.RAM[mario_x]


def marioGetY(emu):
    return emu.fc.fceu.RAM[mario_y]


def go(game):
    total = 0
    emu = fceulib.runGame(game)
    startInputs = hold(0x0, 120) + hold(START | JUMP, 30) + hold(0x0, 150)
    maxHeldFrames = 5
    jumpInputs = [hold(JUMP, t) + hold(0x0, 60)
                  for t in range(1, maxHeldFrames + 1)]
    for m in startInputs:
        total = total + 1
        emu.step(m, 0x0)
    total = total + 1
    emu.step(0x0, 0x0)
    start = VectorBytes()
    print("SAVE")
    emu.save(start)
    print("SAVED")
    # outputImage(emu, "start")

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

    # find: initial velocity, gravity, additive force, max duration (time
    # after which longer button doesn't help)
    model = marioModel
    xget = marioGetX
    yget = marioGetY

    errorsByFrame = []
    netErrorsByState = {}
    for k in model.states:
        netErrorsByState[k] = [0, 0, 0, 0, 0, 0]
    netErrors = [0, 0, 0, 0, 0, 0]

    lastJumpDuration = 0
    boring = False
    dt = 1.0 / 30.0
    for test, jvec in enumerate(jumpInputs):
        if boring:
            print "Boring by test " + str(test)
            break
        print "Params:" + str(model.params)
        print "Error:" + str(netErrors)
        print("LOAD " + str(test))
        emu.load(start)
        startX = xget(emu)
        startY = yget(emu)
        stats = Stats(startX, startY, 5)
        v = model.makeValuation({("x", 0): startX, ("y", 0): startY})
        emuState = "ground"
        print "Start val " + str(v)
        print "Start Y:" + str(startY)
        # across jumps. refine all params and also find:
        # earlyOutClipVel
        # maxButtonDuration
        for (i, m) in enumerate(jvec):
            # TODO: buttons for real
            # TODO: collisions
            lastV = v.copy()
            oldDY = stats.dy.lastVal
            oldSmoothDY = stats.dy.val()
            hereErrors = calcErrorStep(
                model, v, emu,
                xget, yget,
                m,
                stats
            )

            print "New stats Y:" + str(stats.y.val())
            for ei in range(0, 6):
                netErrors[ei] += abs(hereErrors[ei])
                netErrorsByState[v.state][ei] += abs(hereErrors[ei])
            print "Now DY:" + str(stats.dy.lastVal) + " smooth: " + str(stats.dy.val()) + " acc from that: (dsharp) " + str(stats.dy.lastVal - oldDY) + " (dsmooth) " + str(stats.dy.val() - oldSmoothDY)

            nowY = stats.y.lastVal

            # OK, what happened?
            # Let's try to fit our model to the observations.
            # within jump:
            # "gravity": 0,
            # "jumpStartSpeed": 0,
            # "terminalVY": 0,
            # "risingGravity": 0,
            # use vY of first frame to update initial jump V
            # guess. over time this should converge.
            if 1 <= i <= 2:
                model.params["jumpStartSpeed"].update(stats.dy.lastVal)
            # Similarly for other parameters.
            # TODO: Can we learn these without knowing e.g. invariants on each
            # state's dynamics? Or without knowing that some states use
            # accels and others use vels?
            print "DDYs:" + str(stats.ddy.vals) + " " + str(stats.ddy.netVal)
            if emuState == "up":  # rising
                print "Rising AY:" + str(stats.ddy.lastVal)
                model.params["risingGravity"].update(stats.ddy.lastVal)
            elif emuState == "down":  # falling
                print "Falling AY:" + str(stats.ddy.lastVal)
                model.params["gravity"].update(stats.ddy.lastVal)

            # print str(v)
            print "Now Y:" + str(nowY)
            print "Err Here:" + str(hereErrors)

            errorsByFrame.append(hereErrors)

            if emuState == "ground" and nowY < startY:
                emuState = "up"
            elif emuState == "up" and stats.dy.lastVal >= 0 or abs(stats.ddy.val() - stats.ddy.lastVal) > 1.0:
                emuState = "down"
            elif emuState == "down" and stats.y.lastVal == startY:
                emuState = "ground"

            # update the HA forcibly to have the right valuation. we want
            # to learn frame-to-frame differences, so accumulating tons of
            # error isn't interesting.
            if v.state != emuState and lastV.state == emuState:
                v = lastV
                v.timeInState += dt
                v.time += dt
            elif v.state != emuState:
                # todo: take proper transition?
                v.state = emuState
                v.timeInState = 0
            v.variables[("x", 0)] = stats.x.lastVal
            v.variables[("x", 1)] = stats.dx.lastVal
            v.variables[("x", 2)] = stats.ddx.lastVal
            v.variables[("y", 0)] = stats.y.lastVal
            v.variables[("y", 1)] = stats.dy.lastVal
            v.variables[("y", 2)] = stats.ddy.lastVal
            total += 1
            if emuState == "ground" and len(errorsByFrame) > 5:
                if i > lastJumpDuration:
                    model.params["maxButtonDuration"] = max(
                        test * dt,
                        model.params["maxButtonDuration"]
                    )
                    print "New dur: " + str(model.params["maxButtonDuration"])
                else:
                    boring = True
                print "Duration: " + str(i) + "; Last: " + str(lastJumpDuration)
                lastJumpDuration = i
                break
#            if len(errorsByFrame) >= 20:
#                break
    print "Total steps:" + str(total)
    print "Params:" + str(model.params)
    print "Error:" + str(netErrors)
    emu.load(start)
    print "Cur error:" + str(calcError(model, emu, marioGetX, marioGetY, jumpInputs[0])[0])


if __name__ == "__main__":
    go('mario.nes')
