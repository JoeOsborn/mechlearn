import matplotlib.pyplot as plt

import pymc3 as pm
import ppu_dump
import tracking
import pickle

from os.path import basename

import numpy as np
import pandas as pd
import os
# TODO
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
# but also compilation directory = something involving game?
#import theano
#import theano.tensor as T

from fceulib import VectorBytes
import sys
import jumpfinder
import jump_from_tracks
import fceulib
from jumpfinder import marioModel, DT
import copy

from fceu_help import *


def model_y_predictions(m, trials):
    # then do jump trials with horizontal speed and make sure we learn the right weights.
    # then try to learn the clipping for earlyOutClipVel and other discrete
    # velocity updates
    modelYs = []
    for trial, (moves, stats) in enumerate(trials):
        val = m.makeValuation({("x", 0): stats.x.allVals[0],
                               ("y", 0): stats.y.allVals[0]})
        for mi, move in enumerate(moves):
            m.step(
                val,
                DT,
                set(["jump"] if move & jumpButton else []),
                set([("bottom", "ground")]
                    if (val.variables[("y", 0)] >= stats.y.allVals[0] - 1 and
                        mi >= 5)
                    else []))
            if (val.variables[("y", 0)] >= stats.y.allVals[0] - 1 and mi >= 5):
                val.variables[("y", 0)] = stats.y.allVals[0]
            modelYs.append(val.variables[("y", 0)])
            # print mi,val.state,val.variables[("y",0)],val.variables[("y",1)],stats.y.allVals[mi+1]
        modelYs.append(modelYs[-1])
    return modelYs


def graph_model(trials, minHold, maxHold, trace, summaryHA, outputname):
    # Visualize the new approach with a weight vector and clipping.
    # EXECUTE ME after setting up the third model (no resets, no
    # matrix-ization)
    samples = 100
    interesting_trials = []
    if len(trials) <= 3:
        interesting_trials = trials
    else:
        interesting_trials = [trials[0], trials[len(trials)/2], trials[-1]]

    trials = interesting_trials
    realYs = []
    for (_moves, stats) in trials:
        realYs = realYs + stats.y.allVals
    plt.figure(figsize=(20, 10))

    meanYs = model_y_predictions(summaryHA, trials)
    plt.plot(meanYs, 'k-')
    sample = 0
    for rand_trace in np.random.randint(len(trace) * 0.75,
                                        len(trace), samples):
        sample += 1
        t = trace[rand_trace]
        tdict = {}
        for mn in jump_from_tracks.mode_names:
            tdict[mn+"_dx_acc"] = t[mn+"_dx_acc"]
            tdict[mn+"_dy_acc"] = t[mn+"_dy_acc"]
            tdict[mn+"_dx_weights__0"] = t[mn+"_dx_weights"][0]
            tdict[mn+"_dx_weights__1"] = t[mn+"_dx_weights"][1]
            tdict[mn+"_dx_weights__2"] = t[mn+"_dx_weights"][2]
            tdict[mn+"_dy_weights__0"] = t[mn+"_dy_weights"][0]
            tdict[mn+"_dy_weights__1"] = t[mn+"_dy_weights"][1]
            tdict[mn+"_dy_weights__2"] = t[mn+"_dy_weights"][2]
        m = jump_from_tracks.model_params_to_ha(minHold, maxHold, tdict)
        modelYs = model_y_predictions(m, trials)
        plt.plot(modelYs, "r.", alpha=0.1)
    plt.plot(realYs, "kx")
    plt.gca().invert_yaxis()
    plt.savefig(outputname + '.png')
    # plt.show()
    plt.clf()

if __name__ == "__main__":
    pkl = sys.argv[1]
    (rom, start_movie,
     jumpButton,
     trackID, trials,
     minHold, maxHold,
     model, trace,
     ha) = pickle.load(open(pkl))
    outfile = pkl
    graph_model(trials, minHold, maxHold, trace, ha, outfile)
