import numpy as np
from math import log
from unionfind import UnionFind
import networkx as nx
from networkx.algorithms import matching
import scipy.stats


def weight(data, track, R):
    distance = np.linalg.norm(data - track)
    return scipy.stats.norm(0, R).pdf(distance)


def tracks_from_sprite_data(sprite_data):
    # uses normalized pmi as a thresholding mechanism
    # -1 = no co-occurences,
    # 0 = independent
    # 1 = only co-occur

    threshold = 0.1
    # Read in data
    timesteps = {}
    pSprite = {}

    for dat in sprite_data:
        if dat[0] not in timesteps:
            timesteps[dat[0]] = []
        timesteps[dat[0]].append(dat[1:])
        if dat[1] not in pSprite:
            pSprite[dat[1]] = 0
        pSprite[dat[1]] += 1

    # get the per timestep data, as well as total counts for each sprite are,
    # if they are within width and height of each other,
    # add that as a co-occurrence

    all_sprites = {}
    sorted_timesteps = sorted(timesteps)
    for timestep in sorted_timesteps:
        dat = timesteps[timestep]
        for ii in range(len(dat)):
            for jj in range(len(dat)):
                if ii != jj:
                    id1 = dat[ii][0]
                    sprite1 = dat[ii][1]
                    sprite2 = dat[jj][1]
                    dx = sprite1[0] - sprite2[0]
                    dy = sprite1[1] - sprite2[1]
                    height1 = 16 if (sprite1[-1][0] & (1 << 5)) else 8
                    height2 = 16 if (sprite2[-1][0] & (1 << 5)) else 8
                    id2 = (dat[jj][0], dx, dy)
                    if (((height1 != height2) or
                         (abs(dx) != 8 and abs(dx) != 0) or
                         (abs(dy) != height1 and abs(dy) != 0))):
                        continue
                    if id1 not in all_sprites:
                        all_sprites[id1] = {}
                    if id2 not in all_sprites[id1]:
                        all_sprites[id1][id2] = 0
                    all_sprites[id1][id2] += 1

    # now we have all of the pieces and can calculate the pmi for each pair of
    # sprites
    accepted = set()
    for sprite in all_sprites:
        dat = []
        px = float(pSprite[sprite]) / float(len(timesteps))

        for other in all_sprites[sprite]:
            py = float(pSprite[other[0]]) / float(len(timesteps))
            pxy = float(all_sprites[sprite][other]) / float(len(timesteps))
            d = log(px * py) / log(pxy) - 1
            if d > threshold:
                s = sprite
                o = other
                if sprite > other:
                    o = sprite
                    s = other
                accepted.add((s, o))

    timesteps_bb = {}
    for timestep in sorted_timesteps:
        dat = timesteps[timestep]
        blobs = UnionFind()
        for ii in range(len(dat)):
            for jj in range(len(dat)):
                if ii != jj:
                    id1 = dat[ii][0]
                    sprite1 = dat[ii][1]
                    sprite2 = dat[jj][1]
                    dx = sprite1[0] - sprite2[0]
                    dy = sprite1[1] - sprite2[1]
                    id2 = (dat[jj][0], dx, dy)
                    if (id1, id2) in accepted:
                        blobs.union(ii, jj)
        merged = {}
        for ii in range(len(dat)):
            set_id = blobs[ii]
            if set_id not in merged:
                merged[set_id] = set()
            merged[set_id].add(ii)

        bounding_boxes = {}
        timesteps_bb[timestep] = bounding_boxes
        for set_id, sprites in merged.items():
            left = float('inf')
            right = float('-inf')
            top = float('inf')
            bottom = float('-inf')
            for sprite in sprites:
                height = 16 if (dat[sprite][1][-1][0] & (1 << 5)) else 8
                left = min(dat[sprite][1][0], left)
                right = max(dat[sprite][1][0] + height, right)
                top = min(dat[sprite][1][1], top)
                bottom = max(dat[sprite][1][1] + height, bottom)

            bounding_boxes[((left + right) / 2,
                            (top + bottom) / 2,
                            left,
                            right,
                            top,
                            bottom)] = set([dat[ii][0]
                                            for ii in sorted(merged[set_id])])

    sigma = 8.0
    min_gate = 3.0

    tracks = {}
    old_tracks = []
    track_counter = 0
    coast = {}

    max_timestep = sorted_timesteps[-1]
    min_timestep = sorted_timesteps[0]
    for timestep in xrange(min_timestep, max_timestep + 1):
        if timestep not in timesteps_bb:
            to_delete = set()
            for track in tracks:
                if coast[track] > 4:
                    old_tracks.append((track, tracks[track]))
                    to_delete.add(track)
                else:
                    coast[track] += 1
                    tracks[track][timestep] = tracks[track][timestep - 1]

            for track in to_delete:
                del tracks[track]
            continue

        bounding_boxes = timesteps_bb[timestep]
        B = nx.Graph()
        for track in tracks:
            B.add_node(track)
        for sprite_id, sprite in enumerate(bounding_boxes):
            B.add_node('sprite{}'.format(sprite_id))
            B.add_node('track_start{}'.format(sprite_id))
            B.add_edge('sprite{}'.format(sprite_id),
                       'track_start{}'.format(sprite_id),
                       weight=scipy.stats.norm(0, sigma).pdf(min_gate * sigma))

            obs = np.array([sprite[0], 240 - sprite[1]])

            for track in tracks:
                pt = tracks[track][timestep - 1][0]

                pt_weight = weight(obs, pt, 8.0)
                B.add_edge('sprite{}'.format(sprite_id), track,
                           weight=pt_weight)

        match = matching.max_weight_matching(B)
        just_created = set()
        for sprite_id, sprite in enumerate(bounding_boxes):
            obs = np.array([sprite[0], 240 - sprite[1]])
            track = match['sprite{}'.format(sprite_id)]
            if 'start' in track:
                track_counter += 1
                just_created.add('track{}'.format(track_counter))
                tracks['track{}'.format(track_counter)] = {
                    timestep: (obs, sprite, bounding_boxes[sprite])}
                coast['track{}'.format(track_counter)] = 0
            else:
                tracks[track][timestep] = (obs, sprite, bounding_boxes[sprite])
                coast[track] = 0
        to_delete = set()
        for track in tracks:
            if track not in match:
                if track not in just_created:
                    if coast[track] > 4:
                        old_tracks.append((track, tracks[track]))
                        to_delete.add(track)
                    else:
                        coast[track] += 1
                        tracks[track][timestep] = tracks[track][timestep - 1]
        for track in to_delete:
            del tracks[track]
    return (tracks, old_tracks)
