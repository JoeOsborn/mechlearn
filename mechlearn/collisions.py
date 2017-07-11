import numpy as np
import math


def get_collisions(old_tracks, new_tracks,
                   id2colorized,
                   nametables, attributes):
    track_dats = []
    for track in old_tracks:
        trackID = track[0]
        trackDict = track[1]
        trackDat = []
        for t in sorted(trackDict):
            trackDat.append([t] + list(trackDict[t][1]))
        trackDat = np.array(trackDat)
        track_dats.append((trackID, "old", trackDat))
    for trackID, track in new_tracks.items():
        trackDict = track
        trackDat = []
        for t in sorted(trackDict):
            trackDat.append([t] + list(trackDict[t][1]))
        trackDat = np.array(trackDat)
        track_dats.append((trackID, "new", trackDat))
    outputs = {}
    for tid, oldOrNew, exemplarTrack in track_dats:
        exemplarTrack = track
        # <time, center_x, center_y, left,right,top,bottom>

        tile_collisions = {}
        for pt in exemplarTrack:
            time = pt[0]
            nametable = nametables[time]
            attribute = attributes[time]
            left, right, bottom, top = pt[-4:]

            tile_collision = set()

            for nt, attr in zip(nametable[(bottom + 1):(top - 1),
                                          (left + 1):(right - 1)].ravel(),
                                attribute[(bottom + 1):(top - 1),
                                          (left + 1):(right - 1)].ravel()):
                tile_collision.add((nt, attr, 'internal'))

            for nt, attr in zip(nametable[(bottom - 1):(bottom + 1),
                                          (left + 1):(right - 1)].ravel(),
                                attribute[(bottom - 1):(bottom + 1),
                                          (left + 1):(right - 1)].ravel()):
                tile_collision.add((nt, attr, 'bottom'))

            for nt, attr in zip(nametable[(top - 1):(top + 1),
                                          (left + 1):(right - 1)].ravel(),
                                attribute[(top - 1):(top + 1),
                                          (left + 1):(right - 1)].ravel()):
                tile_collision.add((nt, attr, 'top'))

            for nt, attr in zip(nametable[(bottom + 1):(top - 1),
                                          (left - 1):(left + 1)].ravel(),
                                attribute[(bottom + 1):(top - 1),
                                          (left - 1):(left + 1)].ravel()):
                tile_collision.add((nt, attr, 'left'))

            for nt, attr in zip(nametable[(bottom + 1):(top - 1),
                                          (right - 1):(right + 1)].ravel(),
                                attribute[(bottom + 1):(top - 1),
                                          (right - 1):(right + 1)].ravel()):
                tile_collision.add((nt, attr, 'right'))

            tile_collisions[time] = tile_collision

        sprite_collisions = {}
        for track in old_tracks:
            trackID = track[0]

            if trackID != tid:
                trackDict = track[1]
                trackDat = []
                isGood = False
                for sourceT in tile_collisions:
                    if sourceT in trackDict:
                        isGood = True
                        break
                if isGood:
                    overlapping = []
                    for t in sorted(trackDict):
                        trackDat.append([t] + list(trackDict[t][1]))
                        if t in exemplarTrack[:, 0]:
                            overlapping.append(t)
                    trackDat = np.array(trackDat)
                    startOverlapping = overlapping[0]
                    endOverlapping = overlapping[-1]
                    for pt in trackDat:
                        if pt[0] > endOverlapping:
                            break
                        if pt[0] >= startOverlapping:
                            exemplarPt = exemplarTrack[
                                exemplarTrack[:, 0] == pt[0],
                                :][0]
                            if (pt[3] < exemplarPt[4] and
                                pt[4] > exemplarPt[3] and
                                pt[5] < exemplarPt[6] and
                                    pt[6] > exemplarPt[5]):
                                diff = pt[1:3] - exemplarPt[1:3]
                                angle = math.atan2(diff[1],
                                                   diff[0]) * 180 / 3.14159

                                if angle < 45.0 or angle > 315.0:
                                    direction = 'bottom'
                                elif angle < 135.0:
                                    direction = 'right'
                                elif angle < 225.0:
                                    direction = 'top'
                                else:
                                    direction = 'left'
                                if pt[0] not in sprite_collisions:
                                    sprite_collisions[pt[0]] = set()
                                sprite_collisions[pt[0]].add(
                                    (tuple(trackDict[pt[0]][2]), direction))
        all_collisions = {}
        for t in sprite_collisions:
            all_collisions[t] = sprite_collisions[t]
        for t in tile_collisions:
            for coll in tile_collisions[t]:
                if t not in all_collisions:
                    all_collisions[t] = set()
                all_collisions[t].add(((coll[0], coll[1], 'tile'), coll[2]))
        outputs[(tid, oldOrNew)] = all_collisions
    return outputs
