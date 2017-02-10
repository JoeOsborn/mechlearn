
# coding: utf-8

# In[ ]:
import sys
args = sys.argv[1].split(".")[0].split("_")[1:]
print args

import pickle
import numpy as np
import os
os.environ["THEANO_FLAGS"] = (
    "mode=FAST_RUN,device=cpu,floatX=float32,compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s-{}-{}".format(
        args[1],
        args[2])
)
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pymc3 as pm
import math
import networkx as nx
import nxpd


# In[ ]:

tracks = pickle.load(open("mario_tracks.pkl"))


# In[ ]:


tracksToKeep = ['track2']
kept = []


for track in sorted(tracks[1]):
    trackID = track[0]
    trackDict = track[1]
    trackDat = []
    for t in sorted(trackDict):
        trackDat.append([t] + list(trackDict[t][1]))
    trackDat = np.array(trackDat)
    if trackID in tracksToKeep:
        kept.append(trackDat)
    # plt.plot(trackDat[:,0],256-trackDat[:,2])
    # plt.show()

for trackID in sorted(tracks[0]):
    track = tracks[0][trackID]
    trackDict = track
    trackDat = []
    for t in sorted(trackDict):
        trackDat.append([t] + list(trackDict[t][1]))
    trackDat = np.array(trackDat)
    if trackID in tracksToKeep:
        kept.append(trackDat)
    # plt.plot(trackDat[:,0],256-trackDat[:,2])
    # plt.show()
track = np.vstack(kept)
track = track[:500, :]
plt.plot(track[:, 0], 256 - track[:, 2])
plt.show()


# In[ ]:

def thresholds(vals, t_window=5):
    last_diff = 0
    last_diff_t = 0
    max_v = min(vals)
    min_v = max(vals)
    thresholds = set([0, min_v, max_v])
    for t, v in enumerate(vals):
        if (t - last_diff_t) == t_window:
            thresholds.add(last_diff)
        if v != last_diff:
            last_diff = v
            last_diff_t = t
    return thresholds

axis = 2
window = 3

velocities = track[1:, axis] - track[:-1, axis]

thresholds(velocities)


def samey_intervals(vals, t_window=5):
    last = vals[0]
    start = -1
    accum = 0
    intervals = []
    for t, v in enumerate(vals):
        if v != last:
            accum += 1
        if last == v:
            accum = 0
        if accum > 1:
            last = v
            if (t - accum) - start > t_window:
                intervals.append(start + 1)
                intervals.append(t - accum)
                intervals.append(t - accum + 1)
            start = t - accum
            accum = 0
    return intervals


def zero_crossings(vals):

    sign_intervals = []
    last_sign_change = 0
    vsigns = np.sign(vals)
    last_sign = vsigns[0]

    zeros = np.zeros(vsigns.shape)
    zeros[vsigns == 0] = 1
    accum = 0
    for t in range(len(zeros)):
        if zeros[t] == 1:
            accum += 1
        else:
            accum = 0
        zeros[t] = accum
    last = 0
    for t, v in enumerate(zeros):
        if v == 1:
            sign_intervals.append(t)
        elif v == 0 and last > 5:
            sign_intervals.append(t - 1)
        last = v
    return sorted(set(sign_intervals))

# In[ ]:


def samey_intervals_(vals, t_window=5):
    last_diff = 0
    last_diff_t = 0
    intervals = []
    for t, v in enumerate(vals):
        changed = (v != last_diff and (
            t == len(vals) - 1 or vals[t + 1] != last_diff))
        if ((t - last_diff_t) >= t_window) and changed:
            intervals.append(last_diff_t + 1)
            intervals.append(t - 1)
        if changed:
            last_diff = v
            last_diff_t = t - 1
    intervals.append(t - 1)
    return sorted(set(intervals))

axis = 2
window = 2

velocities = track[1:, axis] - track[:-1, axis]
smoothed = scipy.ndimage.filters.convolve1d(
    velocities, np.ones(window) / window)

sameys = samey_intervals(velocities)

plt.plot(velocities[:])
plt.plot(np.array(sameys), velocities[np.array(sameys, dtype='int')], 'rx')
plt.xlim((190, 220))
plt.show()


# In[ ]:

def zero_crossings_(vals):
    sign_intervals = []
    last_sign_change = 0
    vsigns = np.sign(vals)
    last_sign = vsigns[0]
    for t, v in enumerate(vsigns):
        if v == last_sign or (t < len(vsigns) and vsigns[t + 1] == last_sign):
            continue
        elif v != last_sign:
            sign_intervals.append(last_sign_change)
            sign_intervals.append(t - 1)
            sign_intervals.append(t)
            last_sign = v
            last_sign_change = t
    if t - 1 != last_sign_change:
        sign_intervals.append(last_sign_change)
        sign_intervals.append(t)
    return sorted(set(sign_intervals))

axis = 2
window = 2

zeroxs = zero_crossings(velocities)

plt.plot(velocities[:])
plt.plot(np.array(zeroxs), velocities[np.array(zeroxs, dtype='int')], 'rx')
plt.xlim((190, 220))
plt.show()


def jumps(vals):
    return [t for t in range(1, len(vals)) if np.abs(vals[t] - vals[t - 1]) > 1]


# In[ ]:

axis = 2

same_window = 5

switch_points = set(zero_crossings(velocities))

samey_points = set(samey_intervals(velocities, t_window=same_window))

jumps = set(jumps(velocities))

velocity_times = sorted(switch_points |
                        samey_points |
                        jumps |
                        set([0, len(velocities) - 1]))

print len(velocity_times)

plt.plot(track[:, 2])
plt.plot(np.array(velocity_times), track[
         np.array(velocity_times, dtype='int'), 2], 'rx')
plt.show()


# In[ ]:

import fceulib
inputVec = [i for i in fceulib.readInputs('Illustrative.fm2')]


def button_change_times(button_masks):
    last_mask = 0
    mask_times = []
    # Note: for a temporary optimization we could drop certain buttons?
    for t, b in enumerate(button_masks):
        if t < track[0, 0]:
            continue
        if t > track[-1, 0]:
            break
        if b != last_mask:
            mask_times.append(t - track[0, 0])
            last_mask = b
    return mask_times

button_times = button_change_times(inputVec)

plt.plot(np.array(button_times), np.array(button_times) * 0, 'rx')
plt.show()


# In[ ]:

# Templates is an array of model-generating functions of increasing complexity.
templates = [
    # Constant 0 velocity
    ("c0", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=0,
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
    # Constant velocity from old value
    ("cP", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=pv,
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
    # Fixed constant velocity
    ("cN", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=pm.Normal(n + "_N", mu=0, sd=10),
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
    # Constant acceleration from 0
    ("acc0", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=pm.Normal(n + "_acc", mu=0, sd=10) * vs[:, 0],
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
    # Constant acceleration from old velocity value
    ("accP", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=pv + pm.Normal(n + "_acc", mu=0, sd=10) * vs[:, 0],
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
    # Constant acceleration from fixed constant velocity
    ("accN", lambda n, axis, vs, pv: pm.Normal(
        n,
        mu=pm.Normal(n + "_N", mu=0, sd=10) + \
        pm.Normal(n + "_acc", mu=0, sd=20) * vs[:, 0],
        sd=pm.HalfCauchy(n + "_err", beta=10),
        observed=vs[:, axis]
    )),
]
templates = templates
type2ind = {t[0]: i for i, t in enumerate(templates)}
print type2ind


# In[ ]:

iterations = 5000


def model_template_generate(template_i, axis, segment, prev_val):
    axisNames = [None, "x", "y"]
    template = templates[template_i]
    (tn, t) = template
    with pm.Model() as model:
        lik = t(axisNames[axis], axis, segment, prev_val)
        step = pm.Metropolis()
        trace = pm.sample(iterations, step, progressbar=False)
        subtrace = trace[len(trace) / 2:-1:10]
    return (tn, model, subtrace)


# In[ ]:

def model_set_generate(data):
    track, all_times, axis, i, j = data
    t0 = all_times[i]
    t1 = all_times[j]
    print "go", i, j, t0, t1
    if t0 == 0:
        segment = track[t0 + 1:t1 + 1] - track[t0:t1]
        prev_vel = float('inf')
    elif t1 + 1 > np.shape(track)[1]:
        extended_track = np.concatenate((track, [track[-1]]))
        segment = extended_track[t0 + 1:t1 + 1] - extended_track[t0:t1]
        prev_vel = track[t0, axis] - track[t0 - 1, axis]
    else:
        #9,10,11 - 8,9,10
        segment = track[t0 + 1:t1 + 1] - track[t0:t1]
        prev_vel = track[t0, axis] - track[t0 - 1, axis]
    segment[:, 0] = range(0, np.shape(segment)[0])
    result = map(
        lambda ti: model_template_generate(ti,
                                           axis,
                                           segment,
                                           prev_vel),
        range(0, len(templates)))
    return (i, j, t0, t1, result)


# In[ ]:

iterations = 5000
all_times = sorted(set(velocity_times + button_times))

if args[0] == "-1" or args[0] == None:
    end_time = len(all_times)
else:
    end_time = int(args[0])
if len(args) > 1:
    chunk = int(args[1])
    chunks = int(args[2])

all_times = all_times[:end_time]  # [:len(all_times)/4]


# In[ ]:

print "Points:", len(all_times)
print all_times
plt.plot(velocities[:all_times[-1]])
plt.plot(np.array(all_times),
         velocities[np.array(all_times, dtype='int')],
         'rx')
plt.show()

# Takahashi Meijin constant, 60 frames / 16 inputs ~= 4 frames per input.
# But note that in general transitions may happen more frequently due to
# collisions, etc.
min_interval = 4


likes = [None] * len(all_times)
time_per_chunk = int(math.ceil(end_time / float(chunks)))
start_time = time_per_chunk * chunk
this_end_time = min(start_time + time_per_chunk, end_time)

# Janky as heck but a straightforward way to load up models built in parallel

for i in range(start_time, this_end_time):
    print "calc likes[i]", i
    likes[i] = [None] * len(all_times)
    t0 = all_times[i]
    print t0
    min_likelihood = float('inf')
    for j in range(i + 1, len(all_times)):
        # TODO: Use min_interval here?
        js = model_set_generate((track, all_times, axis, i, j))
        # the_templates = js[-1]
        # foundOne = False
        # for tn, mod, trace in the_templates:
        #     dt = float(all_times[j] - all_times[i])
        #     logp = -np.mean([mod.logp(pt)
        #                      for pt in trace]) / dt
        #     print logp
        #     if logp < min_likelihood:
        #         min_likelihood = logp
        #         foundOne = True
        # if not foundOne:
        #    break
        likes[i][j] = js

pickle.dump((start_time, end_time, axis, all_times, track, velocities, inputVec, likes),
            open('mario_{}_{}_{}.like.pkl'.format(end_time, chunk, chunks), 'wb'))
print "Done!"


# In[ ]:
