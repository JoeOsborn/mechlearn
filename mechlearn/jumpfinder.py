import fceulib
from fceulib import VectorBytes
from PIL import Image

"""
Character learning has two components.  First, we need to learn the structure of the automaton representing the automaton.  ProbFOIL, a relational learning library built on top of ProbLog, seems a natural choice---at any rate, it remains for future work.

Once we have a structure, we can fit parameters to that structure.  Example parameters include the dynamics of each state and any parameters on guards.  Since we only consider one character at a time, we can abstract the environment into occluding collisions, non-occluding collisions, and hostile collisions.

We'll work with non-hierarchical, non-concurrent hybrid automata for now.  Also, we'll assume left and right movement are mirrored.  Our representation of a state will define, for each variable, whether it has a constant velocity or a constant acceleration (and what that value is); and what transitions are available under what circumstances (a conjunction of abstracted inputs, continuous variable thresholds, timers, any collisions of any type at any normal to the character, and negations of the same).  A state machine is a set of states and an initial state.
"""


class HA:

    def __init__(self, params, vbls, constraints, states, initial):
        self.params = params
        self.variables = {}
        self.variableNames = set()
        for k, v in vbls.items():
            self.variableNames.add(k)
            self.variables[(k, 0)] = v
            self.variables[(k, 1)] = 0
            self.variables[(k, 2)] = 0
        self.constraints = constraints
        self.states = states
        self.initial = initial

    def makeValuation(self, inits):
        values = self.variables.copy()
        values.update(inits)
        return HAVal(values, self.initial, 0)

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
            x = self.toValue(flows.get((v, 0), 0), val)
            dx = self.toValue(flows.get((v, 1), 0), val)
            ddx = self.toValue(flows.get((v, 2), 0), val)
            oldX = val.variables[(v, 0)]
            oldDX = val.variables[(v, 1)]
            if x != 0:
                oldX = val.variables[(v, 0)]
                oldDX = val.variables[(v, 1)]
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
                val.variables[(v, 0)] += val.variables[(v, 1)]
                self.bound(val, (v, 0))
        val.timeInState += dt
        val.time += dt

    def discreteStep(self, val, transitions, buttons):
        for t in transitions:
            if t.guardSatisfied(self, val, buttons):
                for k, v in (t.update or {}).items():
                    val.variables[k] = self.toValue(v, val)
                if val.state != t.target:
                    val.state = t.target
                    val.timeInState = 0
                break

    def toValue(self, expr, valuation):
        if expr in self.params:
            return self.params[expr]
        elif expr in valuation.variables:
            return valuation.variables[expr]
        elif expr is tuple and expr[0] == "max":
            return max(self.toValue(expr[1], valuation),
                       self.toValue(expr[2], valuation))
        elif expr is tuple and expr[0] == "min":
            return min(self.toValue(expr[1], valuation),
                       self.toValue(expr[2], valuation))
        return expr


class HAVal:

    def __init__(self, vbls, initial, t):
        self.variables = vbls
        self.state = initial
        self.time = t
        self.timeInState = 0


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
    {"gravity": -10,
     "jumpStartSpeed": 20,
     "terminalVY": -40,
     "risingGravity": -5,
     "maxButtonDuration": (1.0 / 30.0),
     "earlyOutClipVel": 10},
    {"x": 0, "y": 0},
    {("y", 1): ("terminalVY", 200)},
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


def calcErrorStep(model, val,
                  emulator, xget, yget,
                  m,
                  lastX, lastY, lastDX, lastDY):
    model.step(val, 1.0 / 30.0, set(["jump"] if m & JUMP else []))
    emulator.step(m, 0x0)
    nowX = xget(emulator)
    nowY = yget(emulator)
    nowDX = nowX - lastX
    nowDY = nowY - lastY
    nowDDX = nowDX - lastDX
    nowDDY = nowDY - lastDY
    hereErrors = (nowX - val.variables[("x", 0)],
                  nowX - val.variables[("y", 0)],
                  nowDX - val.variables[("x", 1)],
                  nowDY - val.variables[("y", 1)],
                  nowDDX - val.variables[("x", 2)],
                  nowDDY - val.variables[("y", 2)])
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
    lastX = startX
    lastY = startY
    lastDX = 0
    lastDY = 0
    for m in actions:
        # TODO: buttons for real
        # TODO: collisions
        hereErrors = calcErrorStep(
            model, v, emulator,
            xget, yget,
            m,
            lastX, lastY, lastDX, lastDY
        )
        for i in range(0, 6):
            netErrors[i] += abs(hereErrors[i])
            netErrorsByState[v.state] += abs(hereErrors[i])
        nowX = xget(emulator)
        nowY = yget(emulator)
        lastDX = nowX - lastX
        lastDY = nowY - lastY
        lastX = nowX
        lastY = nowY
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
    jumpInputs = [hold(JUMP, t) + hold(0x0, 60)
                  for t in range(1, 30)]
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
    for v, jvec in enumerate(jumpInputs):
        print("LOAD " + str(v))
        emu.load(start)
        startX = xget(emu)
        startY = yget(emu)
        v = model.makeValuation({("x", 0): startX, ("y", 0): startY})
        errorsByFrame = []
        netErrorsByState = {}
        for k in model.states:
            netErrorsByState[k] = [0, 0, 0, 0, 0, 0]
        netErrors = [0, 0, 0, 0, 0, 0]
        lastX = startX
        lastY = startY
        lastDX = 0
        lastDY = 0
        for m in jvec:
            # TODO: buttons for real
            # TODO: collisions
            hereErrors = calcErrorStep(
                model, v, emu,
                xget, yget,
                m,
                lastX, lastY, lastDX, lastDY
            )
            for i in range(0, 6):
                netErrors[i] += abs(hereErrors[i])
                netErrorsByState[v.state][i] += abs(hereErrors[i])
            nowX = xget(emu)
            nowY = yget(emu)
            lastDX = nowX - lastX
            lastDY = nowY - lastY
            lastX = nowX
            lastY = nowY
            errorsByFrame.append(hereErrors)
    print "Total steps:" + str(total)
    print "Error:" + str(netErrors)


if __name__ == "__main__":
    go('mario.nes')
