
# coding: utf-8

# In[ ]:

import sys
args = sys.argv[1:]

import pickle
import numpy as np
import os
os.environ["THEANO_FLAGS"] = (
    "mode=FAST_RUN,device=cpu,floatX=float32,compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s-reg-{}-{}".format(
        args[0],
        args[1])
)
import fceulib
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymc3 as pm
import math
import networkx as nx
import nxpd

print "Loading"

limit = int(args[0])
chunks = int(args[1])
chunk_data = []
for chunki in range(0, chunks):
    chunk_data.append(pickle.load(
        open("mario_{}_{}_{}.like.pkl".format(limit, chunki, chunks))
    ))
    print "Loaded {}/{}".format(chunki, chunks)

(start, _cend, axis, all_times, track,
 velocities, inputVec, _likes) = chunk_data[0]
(_lastStart, end_time, _axis, _all_times, _track,
 _velocities, _inputVec, _lastLikes) = chunk_data[-1]

likes = [None] * end_time

# Janky as heck but a straightforward way to load up models built in parallel

for (cstart, cend, axis, all_times, track, velocities, inputVec, clikes) in chunk_data:
    likes[cstart:cend] = clikes[cstart:cend]

print "Loaded"
# In[ ]:

# What is the mode as of just before each switch-point,
# and what is the accumulated cost of the approximation up to that point?
modes = [(0, None)] * (len(all_times))

# modes is offset from likes by 1

ks = {"c0": 1, "cP": 1, "cN": 2, "acc0": 2, "accP": 2, "accN": 3}

cost = 1

# Takahashi Meijin constant, 60 frames / 16 inputs ~= 4 frames per input.
# But note that in general transitions may happen more frequently due to
# collisions, etc.
min_interval = 4

for j in range(1, len(all_times)):
    least = float("inf")
    least_template = None
    print "j", j
    for i in range(0, j):
        # TODO: Use min_interval here?
        data = likes[i][j]
        if not data:
            continue
        dt = data[3] - data[2]
        the_templates = data[-1]
        print "i", i
        for tn, mod, trace in the_templates:
            k = ks[tn]
            summary = pm.df_summary(trace)
            logp = np.mean([mod.logp(pt) for pt in trace])
            # WAIC
            #crit = -pm.stats.waic(model=mod,trace=trace)
            # DIC
            # crit = pm.stats.dic(model=mod,trace=trace)
            # if np.abs(crit) > 1e5:
            #     crit = float('inf')
            # BPIC
            #crit = pm.stats.bpic(model=mod,trace=trace)
            # AICc
            #crit = 2*k - 2 * logp + (2*(k+1)*(k+2))/(dt-k-2)
            # BIC
            crit = math.log(dt) * k - 2 * logp
            # max-likelihood
            #crit = -logp

            #crit = math.log(dt) - logp
            m_prev = modes[i][0]
            # ??
            # crit = summary["mean"]["y_err"]*dt
            here = crit + m_prev + cost
            print i, j, data[2], data[3], tn, logp, summary["mean"]["y_err"], crit, here, least
            if here < least:
                print "update least", here
                least = here
                # prev_i,this_j,t0,t1,name,summary,criterion
                least_template = (i, j, data[2], data[3], tn, summary, crit)
    modes[j] = (least, least_template)

print "------------\nModes\n------------"
print map(lambda m: m[1], modes)


# In[ ]:

def get_path(modes):
    mj = len(modes) - 1
    path = [modes[mj]]
    while mj > 0:
        mj = modes[mj][1][0]
        path.append(modes[mj])
    return list(reversed(path))[1:]

path = get_path(modes)
print "-------\nPath\n-------"
for ii, p in enumerate(path):
    print ii, p[0], '\n', p[1], '\n'


# In[ ]:

"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""


class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


# In[ ]:

cross = {}

for ii, mode in enumerate(path):
    d = {t[0]: t[1:] for t in likes[mode[1][0]][mode[1][1]][4]}
    model, trace = {t[0]: t[1:]
                    for t in likes[mode[1][0]][mode[1][1]][4]}[mode[1][4]]
    for jj, mode2 in enumerate(path):
        model2, trace2 = {t[0]: t[1:] for t in likes[
            mode2[1][0]][mode2[1][1]][4]}[mode2[1][4]]
        if mode2[1][4] == mode[1][4]:
            try:
                # pm.stats.dic(model=model,trace=trace2)
                crit = -np.mean([model.logp(pt) for pt in trace2])
            except:
                crit = float('inf')
        else:
            crit = float('inf')
        cross[(ii, jj)] = crit


# In[ ]:

complexityWeight = 20
unions = UnionFind()
for d in sorted(cross):
    good = True
    for t in [cross[d], cross[(d[0], d[0])], cross[(d[1], d[1])], cross[(d[1], d[0])]]:
        if t == float('inf'):
            good = False
    if not good:
        continue
    joined = min(cross[d] + cross[(d[0], d[0])],
                 cross[(d[1], d[1])] + cross[(d[1], d[0])])

    if (joined < (cross[(d[0], d[0])] + cross[(d[1], d[1])]) + complexityWeight):
        unions.union(d[0], d[1])
merged = {}
for u in unions:
    # print u, unions[u]
    if unions[u] not in merged:
        merged[unions[u]] = set()
    merged[unions[u]].add(u)
print "------\nMerged\n-------"
print "Count:", len(merged)

print "\n".join(map(lambda m: str((m, path[m][1][:5], path[m][1][5]["mean"].to_dict())), merged))


# In[ ]:

iterations = 1000


# def model_sets_generate2(ti, data):
#     track, all_times, axis, i, j = data
#     segments = []
#     for ind in range(len(i)):
#         t0 = all_times[i[ind]]
#         t1 = all_times[j[ind]]
#         if t0 == 0:
#             segment = track[t0 + 1:t1 + 1] - track[t0:t1]
#             prev_vel = 0
#         elif t1 + 1 > np.shape(track)[1]:
#             extended_track = np.concatenate((track, [track[-1]]))
#             segment = extended_track[t0 + 1:t1 + 1] - extended_track[t0:t1]
#             prev_vel = track[t0, axis] - track[t0 - 1, axis]
#         else:
#             #9,10,11 - 8,9,10
#             segment = track[t0 + 1:t1 + 1] - track[t0:t1]
#             prev_vel = track[t0, axis] - track[t0 - 1, axis]
#         segment[:, 0] = range(0, np.shape(segment)[0])
#         segments.append(segment)
#     segment = np.vstack(segments)
#     result = model_template_generate(ti,
#                                      axis,
#                                      segment,
#                                      prev_vel)
#     return (i, j, t0, t1, result)

# for m in merged:
#     print ''
#     if len(merged[m]) > 1:
#         i_ = []
#         j_ = []
#         for t in merged[m]:
#             mode = path[t]
#             dat = likes[mode[1][0]][mode[1][1]]
#             ti = type2ind[mode[1][4]]
#             i_.append(dat[0])
#             j_.append(dat[1])
#         result = model_sets_generate2(ti, (track, all_times, axis, i_, j_))
#         print m, result
#         # pm.summary(result[4][2])
#         # print pm.df_summary(result[4][2])


# In[ ]:

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', "#aa0000",
          "#00aa00", "#0000aa", "#880000", "#008800", "#000088"]
merged2color = {m: i for i, m in enumerate(sorted(merged))}

for m in merged:
    for t in merged[m]:
        plt.plot(np.array(path[t][1][2:4]), np.array(
            [m, m]), colors[merged2color[m]])

plt.plot(velocities[:all_times[-1]])
plt.plot(np.array(all_times), velocities[
         np.array(all_times, dtype='int')], 'rx')
#plt.xlim((200, 220))
plt.plot()
plt.savefig("mode_changes")


# Interestingly, we have "on the ground for a litle bit" and "on the
# ground for longer" as different modes.


# In[ ]:

print "------\nTransitions\n-----"

start_time = 270
transitions = {}
# Edges into [outer] from [inner]
entries_from = {m: {m: [] for m in merged}
                for m in merged}
# Edges into [outer]
entries = {m: [] for m in merged}
for t in range(1, len(path)):
    if t == 0:
        prev = -1
    else:
        prev = unions[t - 1]

    start = path[t][1][2]
    entries_from[unions[t]][prev].append(start)
    entries[unions[t]].append(start)
    transitions[start] = (prev, unions[t])
    print ((path[t][1][0], start), ":",
           prev, "->", unions[t], "\n",
           path[unions[t]][1][4],
           path[unions[t]][1][5]["mean"].to_dict())


# In[ ]:

G = nx.MultiDiGraph()
for tgt, srcs in entries_from.items():
    G.add_node(tgt, label=str(tgt))
    # Let's learn about tgt
    mtype = path[tgt][1][4]
    params = path[tgt][1][5]["mean"].to_dict()
    params["type"] = mtype
    for k, v in sorted(params.items()):
        #         if k == "y_err":
        #             continue
        G.node[tgt]["label"] = (G.node[tgt]["label"] + "\n" + str((k, v)))
    for src, times in srcs.items():
        for t in times:
            G.add_edge(src, tgt, label=t)

G.add_node(-1)
G.add_edge(-1, unions[0])

nxpd.draw(G)

print "Drawn graph"

# In[ ]:

collisions = pickle.load(open('mario_collisions.pkl'))


def button_changes(button_masks):
    last_mask = 0
    mask_times = {}
    for t, b in enumerate(button_masks):
        b_ = int(b)
        buttons = []
        for ii, c in enumerate(list('RLDUTSBA')):
            if b_ & (1 << (7 - ii)):
                buttons.append(c)
        l_ = int(last_mask)
        last_buttons = []
        for ii, c in enumerate(list('RLDUTSBA')):
            if l_ & (1 << (7 - ii)):
                last_buttons.append(c)
        mask_times[t] = (tuple(last_buttons), tuple(buttons))
        last_mask = b

    return mask_times

print "----\nButton Change Times\n----"

button_change_times = button_changes(inputVec)
for t in sorted(button_change_times):
    print t, button_change_times[t]


# In[ ]:

def sign(num):
    if num < 0:
        return -1
    if num > 0:
        return 1
    return 0


def button_diff(btnsA, btnsB):
    return set(btnsA) - set(btnsB)


def button_intersect(btnsA, btnsB):
    return set(btnsA) & set(btnsB)


def button_union(btnsA, btnsB):
    return set(btnsA) | set(btnsB)


def button_preds(button_pairs):
    here_i = set()
    for bp in button_pairs:
        released_i = button_diff(bp[0], bp[1])
        pressed_i = button_diff(bp[1], bp[0])
        held_i = bp[1]
        for ri in released_i:
            here_i.add(("release", ri))
        for ri in pressed_i:
            here_i.add(("press", ri))
        for ri in held_i:
            here_i.add(("hold", ri))
    return list(here_i)


# In[ ]:

preds = [set()] * len(velocities)
for t in range(0, len(velocities)):
    psi = ([button_change_times[start_time + t + i]
            for i in range(0, 1)],
           #  TODO: stopped colliding/started colliding?  That would mean
           #   I could say "started colliding with X on bottom and also zin,-1"
           #   to help find solid things.
           #     ... no... acc,0 should be enough (walking right across solid tiles)
           #     but I should also consider
           #     a more sophisticated notion of collision.
           #      e.g. "bottom" is good but it should be the lowest bottom tile.
           #      how can I get that?  can I get that?
           #      (OTOH, maybe this isn't even necessary if e.g. "touching my feet against sky"
           #        doesn't cause vy=0 as often as "touching my feet against ground" does. so let's be
           #         sure that's surfaced!)
           (collisions.get(start_time + t, set())),
           (velocities[t - 1], velocities[t])
           )
    buttons_i = psi[0]
    here_i = button_preds(buttons_i)
    for coli in psi[1]:
        here_i.append(("col", coli))
    vel0, vel1 = psi[2]
    if vel0 < vel1:
        here_i.append(("acc", 1))
    if vel0 > vel1:
        here_i.append(("acc", -1))
    if vel0 == vel1:
        here_i.append(("acc", 0))
    if vel1 < 0:
        here_i.append(("vel", -1))
    if vel1 > 0:
        here_i.append(("vel", 1))
    if vel1 == 0:
        here_i.append(("vel", 0))
    if vel0 < 0 and vel1 > 0:
        here_i.append(("zc", 1))
    if vel0 > 0 and vel1 < 0:
        here_i.append(("zc", -1))
    if vel0 < 0 and vel1 == 0:
        here_i.append(("zin", 1))
    if vel0 > 0 and vel1 == 0:
        here_i.append(("zin", -1))
    if vel0 == 0 and vel1 < 0:
        here_i.append(("zout", -1))
    if vel0 == 0 and vel1 > 0:
        here_i.append(("zout", 1))
    preds[t] = set(here_i)


# Now, we need to use NPMI to figure out which conditions are likely to be
# important to the learned transitions.

# In[ ]:


def count_cooccurrences(pred_sets, ignored):
    ocs = {}
    coocs = {}
    nice_pred_sets = []
    for pi in range(0, len(pred_sets)):
        here_i = pred_sets[pi]
        here_i = list(set(here_i) - ignored)
        for pred in here_i:
            ocs[pred] = ocs.get(pred, 0) + 1
        for predii in range(0, len(here_i)):
            if here_i[predii] not in coocs:
                coocs[here_i[predii]] = {}
            for predij in range(0, len(here_i)):
                coocs[here_i[predii]][here_i[predij]] = coocs[
                    here_i[predii]].get(here_i[predij], 0) + 1
        nice_pred_sets.append(here_i)
    return nice_pred_sets, ocs, coocs


def calc_npmi(pred_sets, ocs, coocs):
    maximum = float(len(pred_sets))
    npmis = {}
    probs = {}
    # How likely are individual predicates to co-occur
    #  within the transitions to a given target?
    for predx, countx in ocs.items():
        px = countx / maximum
        probs[predx] = px
        for predy, countxy in coocs[predx].items():
            py = ocs[predy] / maximum
            pxy = countxy / maximum
            d = (math.log(px * py) / math.log(pxy) - 1) if pxy != 1 else 1
            npmis[(predx, predy)] = d
    return probs, npmis


def calc_npmi_pred_edge(all_ocs, all_edge_ocs, edge_count, all_edge_count):
    npmis = {}
    probs = {}
    # How likely are individual predicates to co-occur
    #  within the transitions to a given target?
    for predx, countx in all_ocs.items():
        px = countx / float(all_edge_count)
        py = edge_count / float(all_edge_count)
        pxy = all_edge_ocs.get(predx, 0) / float(all_edge_count)
        probs[predx] = all_edge_ocs.get(predx, 0) / float(edge_count)
        assert px <= 1, (px, countx, all_edge_count)
        assert py <= 1, (py, edge_count, all_edge_count)
        assert pxy <= 1, (pxy, all_edge_ocs.get(predx, 0), float(edge_count))
        if pxy == 0:
            d = -1
        elif pxy == 1:
            d = 1
        else:
            d = (math.log(px * py) / math.log(pxy) - 1)
        npmis[predx] = d
    return probs, npmis


def calc_npmi(pred1, pred2, all_counts, counts_by_time):
    norm = float(len(counts_by_time) + 1)
    count1 = all_counts[pred1]
    count2 = all_counts[pred2]
    count12 = 0
    for t, cs in counts_by_time.items():
        count12 += cs.get(pred1, 0) * cs.get(pred2, 0)
    p1 = count1 / norm
    p2 = count2 / norm
    p12 = count12 / norm
    if p12 == 0:
        d = -1
    elif p12 == 1:
        d = 1
    else:
        d = math.log(p1 * p2) / math.log(p12) - 1
    return d


# In[ ]:

def count_events(preds):
    all_counts = {}
    counts_by_time = {}
    for t, ps in enumerate(preds):
        counts_by_time[t] = {}
        for p in ps:
            all_counts[p] = all_counts.get(p, 0) + 1
            counts_by_time[t][p] = counts_by_time[t].get(p, 0) + 1
        if t in transitions:
            tr = transitions[t]
            key = ("tr", tr)
            all_counts[key] = all_counts.get(key, 0) + 1
            counts_by_time[t][key] = counts_by_time[t].get(key, 0) + 1
            (_, dest) = tr
            keystar = ("tr", ("*", dest))
            all_counts[keystar] = all_counts.get(keystar, 0) + 1
            counts_by_time[t][keystar] = counts_by_time[t].get(keystar, 0) + 1
    return all_counts, counts_by_time

all_counts, counts_by_time = count_events(preds)


# In[ ]:

# Let's figure out which tiles block movement on which sides!
# co-occurrence of (col, BLAH) and acc0 for each BLAH.
# cluster together tiles which block on a given side (for now, all those with co-occurrence over threshold)
# then add new preds!

def cond_prob(e1s, e2, all_counts, counts_by_time):
    p2 = all_counts[e2] / float(len(counts_by_time))
    count12 = 0
    for t, cs in counts_by_time.items():
        any_e1_present = False
        for e1 in e1s:
            if e1 in cs:
                any_e1_present = True
                break
        if any_e1_present and (e2 in cs):
            count12 += 1
    p12 = count12 / float(len(counts_by_time))
    return p12 / p2

block_chance = {}
for thing, count in all_counts.items():
    # TODO: generalize back to all sides, but note "colliding on right with something" -> "vely=0" is not that sensible.
    #  need a notion of acc,vel,zin,zout and _other axis_ acc,vel,zin,zout.
    if thing[0] != "col" or thing[1][1] != "bottom":
        continue
    block_chance[thing] = cond_prob([("vel", 0), ("acc", 0)],
                                    thing,
                                    all_counts,
                                    counts_by_time)

merged_by_side = {}
# TODO: generalize back to all sides
for side in ["bottom"]:
    blockings = filter(lambda (col, prob): (col[1][0][0] != "solid" and
                                            col[1][1] == side and
                                            prob > 0.8),
                       block_chance.items())
    merged_by_side[side] = set()
    for bcol, bprob in blockings:
        merged_by_side[side].add(bcol)

print "------\nBlocking tiles\n------"
print block_chance, merged_by_side


# In[ ]:

# Let's add new preds now!
new_preds = [set() for i in range(0, len(preds))]
for t, pset in enumerate(preds):
    for side, equiv in merged_by_side.items():
        found = False
        for pred in pset:
            new_preds[t].add(pred)
            if not found and pred[0] == "col" and pred[1] in equiv:
                pset.append(("col", (("solid", equiv), side)))
                found = True
all_counts, counts_by_time = count_events(new_preds)


# In[ ]:

# Let's calculate NPMI between predicates and transitions!
npmis = {}
for thing, count in all_counts.items():
    if thing[0] == "tr":
        print "tr:", thing, count
        tr = thing[1]
        # Find NPMI with every predicate
        for thing2, count in all_counts.items():
            if thing2[0] == "tr":
                continue
            if tr not in npmis:
                npmis[tr] = {}
            npmis[tr][thing2] = calc_npmi(thing,
                                          thing2,
                                          all_counts,
                                          counts_by_time)
    else:
        pass

print "----------\nNPMI By Transition\n---------"
for tr, prednpmis in npmis.items():
    print ":----"
    print tr
    print ":----"
    for pred, pmi in sorted(prednpmis.items(),
                            lambda a, b: sign(b[1] - a[1])):
        print pred, pmi


# In[ ]:


# In[ ]:

def calc_npmi_chained(e1, e2, e3, all_counts, counts_by_time):
    p1 = all_counts[e1] / float(len(counts_by_time) + 1)
    p2 = all_counts[e2] / float(len(counts_by_time) + 1)
    if p1 == 0:
        assert(False)
        print "p1=0"
        return -1
    if p2 == 0:
        assert(False)
        print "p2=0"
        return -1
    count12 = 0
    for t, cs in counts_by_time.items():
        if (e1 in cs) and (e2 in cs):
            count12 += 1
    p12 = count12 / float(len(counts_by_time) + 1)
    if p12 == 0:
        # Never co-occur, avoid log(0)
        return -1
    pmi12 = math.log(p12 / (p1 * p2))
    #p(event ,  causeB| causeA)/ (p(event) * p(causeB|causeA))
    times_1_and_3_and_2_happen = 0
    times_3_and_2_happen = 0
    times_any_happen = 0
    for t, cs in counts_by_time.items():
        if (e1 in cs) and (e3 in cs) and (e2 in cs):
            times_1_and_3_and_2_happen += 1
        if (e3 in cs) and (e2 in cs):
            times_3_and_2_happen += 1
        if (e1 in cs) or (e2 in cs) or (e3 in cs):
            times_any_happen += 1
    p132 = times_1_and_3_and_2_happen / float(len(counts_by_time) + 1)
    if p132 == 0:
        # Never co-occur, avoid log(0)
        return -1
    p13_2 = (p132 / p2)
    p32 = times_3_and_2_happen / float(len(counts_by_time) + 1)
    p3_2 = p32 / p2
    if p13_2 == 0:
        # Never co-occur, avoid log(0)
        return -1
    elif p3_2 == 0:
        # Never co-occur, avoid log(0)
        return -1
    pmi1_23 = math.log(p13_2 / (p1 * p3_2))
    # normalize by log(p(event;causeA;causeB))??? no...
    # Normalize by self-information!
    return (pmi12 + pmi1_23) / (2 * (-math.log(p132)))

# Let's calculate the NPMI of causal pairs with each transition!
print "---------\nPaired-cause NPMI by transition\n---------"
paired_npmis = {}
for thing, count in all_counts.items():
    if thing[0] == "tr":
        print "tr:", thing, count
        tr = thing[1]
        if tr not in paired_npmis:
            paired_npmis[tr] = {}
        # Find NPMI with every predicate
        for thing2, count2 in all_counts.items():
            if thing2[0] == "tr":
                continue
            for thing3, count3 in all_counts.items():
                if thing3[0] == "tr" or thing3 == thing2:
                    continue
                key = (thing2, thing3)
                paired_npmis[tr][key] = calc_npmi_chained(thing,
                                                          thing2,
                                                          thing3,
                                                          all_counts,
                                                          counts_by_time)
        print "\n".join(
            map(str,
                sorted(filter(
                    lambda (k, v): v > 0,
                    paired_npmis[tr].items()),
                    lambda a, b: sign(b[1] - a[1]))))
    else:
        pass
