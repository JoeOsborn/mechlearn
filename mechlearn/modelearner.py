import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fceulib
from fceulib import VectorBytes
from PIL import Image
import numbers
import math
import numpy as np
import mm_systems as mm
import copy
import em
import mm_svm
from sklearn import svm
import mm_svm

# # For Z3
# import site
# site.addsitedir("/usr/local/lib/python2.7/site-packages")

# import z3

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
        self.continuousStep(val, s.flows, dt, collisions)
        self.discreteStep(val, s.transitions, buttons, collisions)

    def continuousStep(self, val, flows, dt, collisions):
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
        # TODO: Handle collisions by clipping positions and velocities
        val.timeInState += dt
        val.time += dt

    def discreteStep(self, val, transitions, buttons, collisions):
        for t in transitions:
            if t.guardSatisfied(self, val, buttons, collisions):
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

    def guardSatisfied(self, m, val, buttons, collisions):
        for g in self.guard:
            return self.primitiveGuardSatisfied(m, g, val, buttons, collisions)

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
    {"gravity": Stat(0.0, None),
     "jumpStartSpeed": Stat(0.0, None),
     "terminalVY": Stat(100000.0, None),
     "risingGravity": Stat(0.0, None),
     "maxButtonDuration": Stat(100000.0, None),
     "minButtonDuration": Stat(0.0, None),
     "earlyOutClipVel": Stat(-100000.0, "interesting_mode"),
     # Just for curiosity/info
     "longestJump": Stat(0, None)},
    {"x": 0, "y": 0},
    {("y", 1): (-1000000.0, "terminalVY")},
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
            # This edge may not always be present.
            HATransition("down",
                         [("colliding", "top", "ground")],
                         {("y", 1): 0}),
            HATransition("down",
                         [("timer", "maxButtonDuration")],
                         None),
            HATransition("down",
                         [("button", "off", "jump"),
                          ("timer", "minButtonDuration")],
                         # Use _max_ here because negative Y is up, positive Y
                         # is down.  The point of this is to bring Y closer to
                         # downwards when you leave up.
                         {("y", 1): ("max", ("y", 1), "earlyOutClipVel")})
        ]),
        "down": HAState({("y", 2): "gravity"}, [
            # this edge may not always be present.
            # It's here too for the case where we are in down state
            # but our velocity is still positive.
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


def runTrials(emu, start, getx, gety, jumpButton):
    # FIXME: Quite domain specific
    maxHeldFrames = 120
    jumpInputs = [hold(jumpButton, t) + hold(0x0, 600)
                  for t in range(1, maxHeldFrames + 1)]
    startX = getx(emu)
    startY = gety(emu)
    trials = []
    length = 0
    for (i, jvec) in enumerate(jumpInputs):
        stats = Stats(startX, startY, 5)
        for (j, move) in enumerate(jvec):
            emu.step(move, 0x0)
            nowX = getx(emu)
            nowY = gety(emu)
            # can also aggregate other useful stuff into stats later
            stats.update(nowX, nowY)
            if gety(emu) == startY and j > 5:
                break
        if j == length:
            break
        length = j
        trials.append((jvec[0:j], stats))
        plt.figure(1)
        plt.plot(stats.y.allVals, '+-')
        plt.figure(2)
        plt.gca().invert_yaxis()
        plt.plot(stats.dy.allVals, 'x-')
        plt.savefig('trials/dys' + str(i))
        plt.close(2)
        emu.load(start)
    plt.figure(1)
    plt.title('Yvals')
    plt.gca().invert_yaxis()
    plt.savefig('trials/ys')
    plt.close(1)
    return trials


def fitJumpModel(emu, start, getx, gety, jumpButton, model):
    trials = runTrials(emu, start, getx, gety, jumpButton)
    # Run EM with the trials as our data.  Refine the model until it does well
    # on all the trials.
    print str(len(trials)) + " trials."
    for i in range(0, 10):
        for j, (moves, stats) in enumerate(trials):
            val = model.makeValuation(
                {("x", 0): stats.x.allVals[0],
                 ("y", 0): stats.y.allVals[0]}
            )
            closs = [0, 0]
            print str(len(moves)) + " moves, " + str(len(stats.y.allVals)) + " measurements"
            for k, m in enumerate(moves):
                x = stats.x.allVals[k + 1]
                y = stats.y.allVals[k + 1]
                model.step(val,
                           DT,
                           set(["jump"] if m & jumpButton else []),
                           # externalized collision theory
                           set([("bottom", "ground")] if abs(val.variables[("y", 0)] - stats.y.allVals[0]) < 0.1 and k >= 5 else []))
                closs[0] += x + 0.5 - model.variables[("x", 0)]
                closs[1] += y + 0.5 - model.variables[("y", 0)]
            plt.plot([i * len(trials) + j], [closs[1]])
    plt.savefig('loss')
    plt.close()


def jump_data(emu, start, getx, gety, jumpButton, sigmaLik=0.9, sigma=0.05):
    # Starts at the origin with zero velocity
    X = np.array([getx(emu), 0, gety(emu), 0])
    # print "Begin:" + str(gety(emu))

    trials = runTrials(emu, start, getx, gety, jumpButton)
    # X = [x x' y y'] char position

    data = mm.MMData()

    # State vector samples
    data.x = X

    # Model samples
    data.num_models = 3
    GROUND = 0
    RISE = 1
    FALL = 2

    data.m = np.array([GROUND])

    data.Qm = [(sigmaLik**2) * np.identity(4),
               (sigmaLik**2) * np.identity(4),
               (sigmaLik**2) * np.identity(4)]

    for (moves, stats) in trials[5:8]:
        mode = GROUND
        data.x = np.vstack(
            (data.x, np.array([stats.x.allVals[0], 0, stats.y.allVals[0], 0])))
        data.m = np.hstack((data.m, mode))
        # print "start:" + str(stats.y.allVals[0])
        for i, m in enumerate(moves):
            # print str(stats.y.allVals[i + 1]) + " from " +
            # str(stats.y.allVals[i])

            data.x = np.vstack((data.x,
                                np.array([stats.x.allVals[i + 1],
                                          stats.x.allVals[
                                              i + 1] - stats.x.allVals[i],
                                          stats.y.allVals[i + 1],
                                          stats.y.allVals[i + 1] - stats.y.allVals[i]])))
            if mode == GROUND and m & jumpButton:
                mode = RISE
            elif mode == RISE and not (m & jumpButton):
                mode = FALL
            # elif mode == FALL and abs(stats.y.allVals[i + 1] -
            #                           stats.y.allVals[0]) <= 1:
            #     mode = GROUND
            data.m = np.hstack((data.m, mode))

    noisy_data = copy.deepcopy(data)
    noisy_data.x += np.random.randn(*noisy_data.x.shape) * sigma

    return data, noisy_data


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

    # SVM parameters
    svm_mis_penalty = 1000000
    # number of modes
    K = 3
    # data with correct mode assignments
    [data, noisy_data] = jump_data(emu, start, getx, gety, jumpButton)
    # Same data, but with (later!) fully random mode assignments
    dataCopy = copy.deepcopy(data)
    # for i in range(len(dataCopy.m)):
    #    dataCopy.m[i] = np.random.randint(K, size=1)

    # Stores the experimental data
    system = mm.system_from_data(dataCopy)
    # Estimates the activation regions using (later!) heavily corrupted data
    data_list = mm_svm.svm_split_data(dataCopy, K)
    classifier_list = mm_svm.svm_fit(data_list, K, mis_penalty=svm_mis_penalty)
    em.init_model(system, dataCopy, classifier_list,
                  normalize_lik=True, scale_prob=False)
    [system,
     dataCopy,
     numSteps,
     avgExTime] = em.learn_model_from_data(system,
                                           dataCopy,
                                           data,
                                           False,
                                           svm_mis_penalty)
    simData = mm.simulate_mm_system(system)
    plt.figure()
    plt.plot(data.x[:, 0], data.x[:, 2], 'b.', label='True data')
    plt.plot(simData.x[:, 0], simData.x[:, 2], 'r-', label='Sim. data')
    plt.legend()
    plt.savefig("simsys", block=False)
    # fitJumpModel(emu, start, getx, gety, jumpButton, marioModel)

    # maxJump = marioModel.params["maxButtonDuration"].val()
    # testInputs = (hold(jumpButton, int(maxJump / DT) + 1) +
    #              hold(0x0, int(0.8 / DT)))
    # emu.load(start)
    # TODO: see if error is improved by using either guessedWhatever or the frame by frame Whatever (is this the same as closed vs open loop?)
    # TODO: Analyze the errors.  First off, it looks like model.y is wrong from "down" onwards.  It also never quite gets as high as the real emulator does in Y.
    #  It looks like there's also weirdness (expected weirdness?) around landing. I think if I told the model when y=startY as a collision, we would be well good. A position error of 51 seems pretty wrong
    #  ... rising accel is not too far off (compared to an estimate of r=g=300 it's still too strong I think?). falling accel is way overestimated (and in fact is not seemingly that great for down either....?). why?
    # (net, gross, byStep, byState) = calcError(marioModel,
    #                                           emu, getx, gety,
    #                                           jumpButton, testInputs)
    # print "Gross error:" + str(gross)
    # print "Net error:" + str(net)
    # print "By state:" + str(byState)
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
