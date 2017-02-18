import pymc3 as pm

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# TODO os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
# but also compilation directory = something involving game?
import theano
import theano.tensor as T

import sys
import jumpfinder
import fceulib
from jumpfinder import marioModel, DT
import copy


def hold(mask, duration):
    return [mask for i in range(duration)]


def jump_seqs(minHeldFrames=1, maxHeldFrames=120, button=0x01):
    return [(t, hold(jumpButton, t) + hold(0x0, 600))
            for t in range(minHeldFrames, maxHeldFrames + 1)]


mode_names = ["ground", "up-control", "up-fixed", "down"]


def record_run(modes, state, t, vbls, all_vbls):
    frame = pd.DataFrame(vbls)
    # print str(t)+":"+str(all_vbls)
    prev_vbls = dict(x=all_vbls["x"][t],
                     y=all_vbls["y"][t],
                     dx=all_vbls["dx"][t-1 if t > 0 else 0],
                     dy=all_vbls["dy"][t-1 if t > 0 else 0])
    modes[state].append((t, prev_vbls, frame))


def generate_labeled_data(allTrials, minHold, maxHold, jumpButton):
    modes = {m: [] for m in mode_names}
    # Let's learn three linear-or-constant velocity models.
    switch_conditions = {
        "ground": {
            # starting to rise
            "up-control": lambda moves, movei, stats: stats.y.allVals[movei+1] != stats.y.allVals[0]
        },
        "up-control": {
            # released button or reached max time
            # TODO: replace movei with "time since jump pressed" by scanning back in moves
            "up-fixed":
            lambda moves, movei, stats: (movei >= minHold and 
                                         (not (moves[movei] and jumpButton))) or movei >= maxHold#,
            # TODO: or hit ceiling
            #"down":lambda moves, movei, stats: stats.y.allVals[movei+1] >= stats.y.allVals[movei]
        },
        "up-fixed": {
            # TODO: or hit ceiling
            "down": lambda moves, movei, stats: stats.y.allVals[movei+1] > stats.y.allVals[movei]
        },
        "down": {
            "ground": lambda moves, movei, stats: stats.y.allVals[movei+1] >= stats.y.allVals[0]
        }
    } if minHold != maxHold else {
        "ground": {
            # starting to rise
            "up-fixed": lambda moves, movei, stats: stats.y.allVals[movei+1] != stats.y.allVals[0]
        },
        "up-fixed": {
            # TODO: or hit ceiling
            "down": lambda moves, movei, stats: stats.y.allVals[movei+1] > stats.y.allVals[movei]
        },
        "down": {
            "ground": lambda moves, movei, stats: stats.y.allVals[movei+1] >= stats.y.allVals[0]
        }
    }

    t = 0
    state_change_t = 0 # start at 1 just so state_change_t - 1 doesn't wrap
    trials = allTrials
    all_vbls = dict(x=[],y=[],dx=[],dy=[],t=[])
    # Tweak the range and increment to get more precise/slower fitting.
    for moves, stats in trials:
        state = "ground"
        vbls = dict(x=[],y=[],dx=[],dy=[],t=[])
        start_t = t
        for i in range(10):
            vbls["x"].append(stats.x.allVals[0])
            all_vbls["x"].append(vbls["x"][-1])
            vbls["y"].append(stats.y.allVals[0])
            all_vbls["y"].append(vbls["y"][-1])
            vbls["dx"].append(0.)
            all_vbls["dx"].append(vbls["dx"][-1])
            vbls["dy"].append(0.)
            all_vbls["dy"].append(vbls["dy"][-1])
            vbls["t"].append(t-state_change_t)
            all_vbls["t"].append(vbls["t"][-1])
            t += 1
        print "Move count:"+str(len(moves))+" min hold:"+str(minHold)
        for i, m in enumerate(moves):
            transitions = switch_conditions[state]
            for target, condition in transitions.items():
                if condition(moves, i, stats):
                    print "Record "+state+"->"+target,state_change_t,t,"prev dy",str(all_vbls["dy"][state_change_t-1])
                    record_run(modes, state, state_change_t, vbls, all_vbls)
                    state_change_t = t
                    vbls["x"] = []
                    vbls["y"] = []
                    vbls["dx"] = []
                    vbls["dy"] = []
                    vbls["t"] = []
                    state = target
                    break
            vbls["x"].append(stats.x.allVals[i+1])
            all_vbls["x"].append(vbls["x"][-1])
            vbls["y"].append(stats.y.allVals[i+1])
            all_vbls["y"].append(vbls["y"][-1])
            vbls["dx"].append(stats.x.allVals[i+1] - stats.x.allVals[i])
            all_vbls["dx"].append(vbls["dx"][-1])
            vbls["dy"].append(stats.y.allVals[i+1] - stats.y.allVals[i])
            all_vbls["dy"].append(vbls["dy"][-1])
            vbls["t"].append(t-state_change_t)
            all_vbls["t"].append(vbls["t"][-1])
            if t < 20:
                print t,all_vbls["t"],(m&jumpButton),i,(minHold-1)
            t += 1
            if state == "ground" and i > 5:
                break
        # Force to ground state. Collections will be reset at the front of the next loop.
        if state != "ground":
            record_run(modes, 
                       state,
                       state_change_t,
                       vbls,
                       all_vbls)
            state_change_t = t


def fit_model(modes):
    # This is the version without matrix-ization, it takes about as long to compile (unmeasured) and samples over 10x faster even single-threaded
    # MAYBE EXECUTE ME! Then the last cell.
    with pm.Model() as modelLinearAndClip:
        accs = dict()
        sigs = dict()
        prev_weights = dict()
        initcliplo = dict()
        initcliphi = dict()
        for m in modes:
            if len(modes[m]) == 0:
                print "Warning, no witness states for mode "+m
                continue
            sigs[m] = dict()
            accs[m] = dict()
            prev_weights[m] = dict()
            initcliplo[m] = dict()
            initcliphi[m] = dict()
            for v in ["dx","dy"]:
                sigs[m][v] = pm.HalfCauchy(m+"_"+v+"_sigma",beta=10.,testval=1.)
                accs[m][v] = pm.Normal(m+"_"+v+"_acc",mu=0,sd=10.,testval=0.)
                allVals = []
                allTs = []
                allPrevs = []
                for _s,prevs,vbls in modes[m]:
                    allVals = allVals + list(vbls[v])
                    allTs = allTs + list(vbls["t"])
                    # TODO: use vars uniformly distributed between prevs[v] and prevs[v]+1 to account for uncertainty?
                    #  Not sure how to get a nice Theano-friendly setup for that though
                    allPrevs = allPrevs + [[prevs["dx"],
                                            prevs["dy"],
                                            1.0] for i in range(len(vbls[v]))]
                init_weights = [1.0 if vi == v else 0.0 for vi in ["dx","dy","$const"]]
                prev_weights[m][v] = pm.Normal(m+"_"+v+"_weights",
                                               mu=init_weights,
                                               testval=init_weights,
                                               sd=10.,
                                               shape=3)
                vals = np.array(allVals)
                # gotta add 1 to every t since by the time we read the values it's been in the state for one timestep already.
                ts = np.array(allTs)+1 
                prevs = np.array(allPrevs)
                muInit = prev_weights[m][v].dot(prevs.T)
                mu = muInit + accs[m][v]*ts
                lik = pm.Normal(m+"_"+v, 
                                mu=mu, 
                                sd=sigs[m][v], 
                                observed=vals)
        print "Find start"
        start = dict()
        #start = pm.approx_hessian(model.test_point)
        print str(start)
        print "Set up step method"
        step = pm.Metropolis()
        #step = pm.NUTS(scaling=start)
        print "Start sampling"
        traceLinearAndClip = pm.sample(10000, step, progressbar=True)
        print "Done!"
    return (modelLinearAndClip, traceLinearAndClip)


def test_model(trials, minHold, maxHold, traceLinearAndClip):
    # Visualize the new approach with a weight vector and clipping.
    # EXECUTE ME after setting up the third model (no resets, no matrix-ization)
    samples = 200
    m = copy.deepcopy(marioModel)

    realYs = []
    for (_moves,stats) in trials:
        realYs = realYs + stats.y.allVals
    plt.figure(figsize=(20,10))
    sample = 0
    for rand_trace in np.random.randint(len(traceLinearAndClip)*0.75, len(traceLinearAndClip), samples):
        sample += 1
        t = traceLinearAndClip[rand_trace]
        m.params["gravity"].update(t["down_dy_acc"]/(DT*DT),True)
        if minHold != maxHold:
            m.params["up-control-gravity"].update(t["up-control_dy_acc"]/(DT*DT),True)
        else:
            # We didn't learn anything for up-control, so use up-fixed instead.
            m.params["up-control-gravity"].update(t["up-fixed_dy_acc"]/(DT*DT),True)
        m.params["up-fixed-gravity"].update(t["up-fixed_dy_acc"]/(DT*DT),True)
    #     if t["up-fixed_dy_acc"] < 0 or t["up-control_dy_acc"] < 0 or t["down_dy_acc"] > 0:
    #         # Outlier, ignore
    #         continue
        m.params["minButtonDuration"].update(minHold*DT,True)
        m.params["maxButtonDuration"].update(maxHold*DT,True)

        if minHold != maxHold:
            m.params["groundToUpControlDYReset"] = (
                "+", 
                t["up-control_dy_weights"][2]/DT,
                ("+", 
                 ("*", t["up-control_dy_weights"][0], ("x", 1)),
                 ("*", t["up-control_dy_weights"][1], ("y", 1)))
            )
            m.params["upControlToUpFixedDYReset"] = (
                "+", 
                t["up-fixed_dy_weights"][2]/DT,
                ("+", 
                 ("*", t["up-fixed_dy_weights"][0], ("x", 1)),
                 ("*", t["up-fixed_dy_weights"][1], ("y", 1)))
            )
        else:
            # Some hacking around the HA structure
            m.params["groundToUpControlDYReset"] = (
                "+", 
                t["up-fixed_dy_weights"][2]/DT,
                ("+", 
                 ("*", t["up-fixed_dy_weights"][0], ("x", 1)),
                 ("*", t["up-fixed_dy_weights"][1], ("y", 1)))
            )
            # Treat up-fixed as a continuation of up-control
            m.params["upControlToUpFixedDYReset"] = ("y", 1)

        # Unused learned weights: ground, down.


        # then do jump trials with horizontal speed and make sure we learn the right weights.
        # then try to learn the clipping for earlyOutClipVel and other discrete velocity updates
        modelYs = []
        modelModes = []
        mode_nums = {"ground": -200, "up-control": -190, "up-fixed": -180, "down": -170}
        for trial, (moves, stats) in enumerate(trials):
            val = m.makeValuation({("x", 0): stats.x.allVals[0],
                                   ("y", 0): stats.y.allVals[0]})
            for mi, move in enumerate(moves):
                m.step(
                    val,
                    DT,
                    set(["jump"] if move & jumpButton else []),
                    set([("bottom", "ground")]
                        if (val.variables[("y", 0)] > stats.y.allVals[0]-1 and
                            mi >= 5)
                        else []))
                modelModes.append(mode_nums[val.state])
                modelYs.append(val.variables[("y",0)])
                # print mi,val.state,val.variables[("y",0)],val.variables[("y",1)],stats.y.allVals[mi+1]
            modelYs.append(modelYs[-1])
            modelModes.append(mode_nums[val.state])
    #     plt.plot(modelYs,"x-",lw=0.6)
    #     plt.plot(modelModes,"+-",lw=0.6)
    # plt.plot(realYs,"o")
    # plt.gca().invert_yaxis()
    # plt.show()


if __name__ == "main":
    game = sys.argv[1]
    jumpButton = sys.argv[2]
    min_len = int(sys.argv[3])
    max_len = int(sys.argv[4])
    # TODO: figure it out dynamically later
    track_id = sys.argv[5]
    # TODO: set up game via start movie
    # emu = fceulib.runGame(game+".nes")
    # startInputs = game+"_start.fm2"

    # for m in startInputs:
    #     emu.step(m, 0x0)
    # start = fceulib.VectorBytes()
    # emu.save(start)

    # TODO: get jump episodes
    # TODO: ppu-drive each jump episode
    # TODO: get tracking info for each jump episode
    # TODO: get collision info for each jump episode? if bottom collisions go back to initial bottom collisions we're done
    # TODO: !! determine minHold, maxHold by looking at episodes and seeing how long we're in the air
    # TODO: fit
    # TODO: test
    # TODO: output as HA description
