import numpy as np
from math import log
from unionfind import UnionFind
import networkx as nx
from networkx.algorithms import matching
import scipy.stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Sequence, List, Set, Optional, TypeVar
from ppu_dump import SpriteIdx, HwSprite, Col, Row, ColRow, Time, SpriteId, SpriteData, DX, DY


SpriteTrack = Tuple[SpriteIdx, HwSprite]

BoxId = Tuple[Col, Row, Col, Col, Row, Row, int, int]  # cx cy l r t b w h
BoundingBoxes = Dict[BoxId, List[SpriteTrack]]

TrackDatum = Tuple[ColRow, BoxId, List[SpriteTrack]]
Track = Dict[Time, TrackDatum]
Tracks = Dict[SpriteId, Track]
IdType1 = TypeVar("IdType1")
IdType2 = TypeVar("IdType2")
Count = int
Cooccur = Dict[IdType1, Dict[IdType2, Count]]

OffsetSpriteIdx = Tuple[SpriteIdx, DX, DY]
OldTracks = List[Tuple[SpriteId, Track]]


def weight(data: ColRow, track: ColRow, R: float) -> float:
    distance = np.linalg.norm(np.array(data) - np.array(track))
    return scipy.stats.norm(0, R).pdf(distance)


def tracks_from_sprite_data(
        sprite_data: SpriteData,
        sigma=8.0,
        min_gate=5.0) -> Tuple[Tracks, OldTracks]:
    # uses normalized pmi as a thresholding mechanism
    # -1 = no co-occurences,
    # 0 = independent
    # 1 = only co-occur

    threshold = 0.01
    # Read in data
    timesteps: Dict[Time, List[SpriteTrack]] = {}
    pSprite: Dict[SpriteIdx, int] = {}

    for tt in sprite_data:
        timesteps[tt] = []
        for sprite in sprite_data[tt]:
            timesteps[tt].append((sprite[1], sprite[2]))
            assert len(sprite) == 3
            if sprite[1] not in pSprite:
                pSprite[sprite[1]] = 0
            pSprite[sprite[1]] += 1

    # get the per timestep data, as well as total counts for each sprite are,
    # if they are within width and height of each other,
    # add that as a co-occurrence
    #          0     1   2    3    4    5    6    7    8    9    10   11   12   13   14   15   16
#    colors:List[str] = ['rx', 'gx', 'bx', 'kx', 'cx', 'mx', 'yx', 'r+', 'g+', 'b+', 'k+', 'c+', 'm+', 'y+', 'rv', 'gv', 'bv', 'kv', 'cv', 'mv', 'yv', 'r^', 'g^', 'b^', 'k^', 'c^', 'm^', 'y^', 'r.', 'g.', 'b.', 'k.', 'c.', 'm.', 'y.']
    # print '\n'.join(['{}: {}'.format(ii,c) for ii,c in enumerate(colors)])
    all_sprites: Cooccur[SpriteIdx, OffsetSpriteIdx] = {}
    sorted_timesteps: Sequence[int] = sorted(timesteps)
    for timestep in sorted_timesteps:
        dat: List[SpriteTrack] = timesteps[timestep]
        for ii in range(len(dat)):
            plt.plot(timestep, dat[ii][1][1], '.')
            for jj in range(len(dat)):
                if ii != jj:
                    id1: SpriteIdx = dat[ii][0]
                    sprite1: HwSprite = dat[ii][1]
                    sprite2: HwSprite = dat[jj][1]
                    dx: DX = sprite1[0] - sprite2[0]
                    dy: DY = sprite1[1] - sprite2[1]
                    height1: int = 16 if (sprite1[-1][0] & (1 << 5)) else 8
                    height2: int = 16 if (sprite2[-1][0] & (1 << 5)) else 8
                    id2: OffsetSpriteIdx = (dat[jj][0], dx, dy)

                    # print id1, dat[jj][0], height1, height2, dx, dy
                    if (((height1 != height2)
                         or (abs(dx) > 8)
                         or (abs(dy) > height1))):
                        continue
                    if id1 not in all_sprites:
                        all_sprites[id1] = {}
                    if id2 not in all_sprites[id1]:
                        all_sprites[id1][id2] = 0
                    all_sprites[id1][id2] += 1
    plt.show()
    plt.clf()
    dat[:] = []

    # now we have all of the pieces and can calculate the pmi for each pair of
    # sprites
    accepted: Set[Tuple[SpriteIdx, SpriteIdx]] = set()
    for sprite_idx in all_sprites:
        px = float(pSprite[sprite_idx]) / float(len(timesteps))

        for other in all_sprites[sprite_idx]:
            py = float(pSprite[other[0]]) / float(len(timesteps))
            pxy = float(all_sprites[sprite_idx][other]) / float(len(timesteps))
            if pxy == 1:
                s = sprite_idx
                o = other[0]
                if s > o:
                    o = sprite_idx
                    s = other[0]
                accepted.add((s, o))
            else:
                d = log(pxy / (px * py)) / -log(pxy)  # log(px * py) / log(pxy) - 1
                # print sprite_idx, other, px, py, pxy, d
                if d > threshold:
                    s = sprite_idx
                    o = other[0]
                    if s > o:
                        o = sprite_idx
                        s = other[0]
                    accepted.add((s, o))
    timesteps_bb: Dict[Time, BoundingBoxes] = {}
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
        # This int is not a sprite index in the global sense, just locally
        merged: Dict[SpriteIdx, Set[SpriteIdx]] = {}
        for ii in range(len(dat)):
            set_id = blobs[ii]
            if set_id not in merged:
                merged[set_id] = set()
            merged[set_id].add(ii)

        bounding_boxes: BoundingBoxes = {}
        timesteps_bb[timestep] = bounding_boxes
        for set_id, sprite_idxs in merged.items():
            left: Optional[int] = None
            right: Optional[int] = None
            top: Optional[int] = None
            bottom: Optional[int] = None
            # print timestep, set_id
            for sprite_idx in sprite_idxs:
                height = 16 if (dat[sprite_idx][1][-1][0] & (1 << 5)) else 8
                sleft = dat[sprite_idx][1][0]
                left = min(sleft, left if left is not None else sleft)
                sright = dat[sprite_idx][1][0] + height
                right = max(sright, right if right is not None else sright)
                stop = dat[sprite_idx][1][1]
                top = min(stop, top if top is not None else stop)
                sbot = dat[sprite_idx][1][1] + height
                bottom = max(sbot, bottom if bottom is not None else sbot)
                # print '\t', top, bottom
            assert left is not None
            assert right is not None
            assert top is not None
            assert bottom is not None
            bounding_boxes[((left + right) // 2,
                            (top + bottom) // 2,
                            left,
                            right,
                            top,
                            bottom,
                            right - left,
                            bottom - top)] = [dat[ii]
                                              for ii in sorted(merged[set_id])]

    tracks: Tracks = {}
    old_tracks: OldTracks = []
    track_counter: int = 0
    coast: Dict[SpriteId, int] = {}

    max_timestep = sorted_timesteps[-1]
    min_timestep = sorted_timesteps[0]
    for timestep in range(min_timestep, max_timestep + 1):
        if timestep not in timesteps_bb:
            to_delete: Set[SpriteId] = set()
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
        for sprite_id, sprite_box in enumerate(bounding_boxes):

            B.add_node('sprite{}'.format(sprite_id))
            B.add_node('track_start{}'.format(sprite_id))
            B.add_edge('sprite{}'.format(sprite_id),
                       'track_start{}'.format(sprite_id),
                       weight=scipy.stats.norm(0, sigma).pdf(min_gate * sigma))

            obs: ColRow = (sprite_box[0], 240 - sprite_box[1])
            # plt.plot(timestep,obs[1],'.')

            for track in tracks:
                pt = tracks[track][timestep - 1][0]

                pt_weight = weight(obs, pt, 8.0)
                B.add_edge('sprite{}'.format(sprite_id), track,
                           weight=pt_weight)

        match = matching.max_weight_matching(B)
        just_created: Set[SpriteId] = set()
        matched_tracks: Set[SpriteId] = set()
        for sprite_id, sprite_box in enumerate(bounding_boxes):
            obs = (sprite_box[0], 240 - sprite_box[1])
            track_key: Optional[SpriteId] = None
            key: SpriteId = 'sprite{}'.format(sprite_id)
            for (a, b) in match:
                if a == key:
                    track_key = b
                    break
                if b == key:
                    track_key = a
                    break
            assert track_key is not None
            matched_tracks.add(track_key)
            if 'start' in track_key:
                track_counter += 1
                just_created.add('track{}'.format(track_counter))
                tracks['track{}'.format(track_counter)] = {
                    timestep: (obs, sprite_box, bounding_boxes[sprite_box])}
                coast['track{}'.format(track_counter)] = 0
            else:
                tracks[track_key][timestep] = (obs, sprite_box, bounding_boxes[sprite_box])
                coast[track_key] = 0
        to_delete = set()
        for track in tracks:
            if track not in matched_tracks:
                if track not in just_created:
                    if coast[track] > 5:
                        old_tracks.append((track, tracks[track]))
                        to_delete.add(track)
                    else:
                        coast[track] += 1
                        tracks[track][timestep] = tracks[track][timestep - 1]
        for track in to_delete:
            del tracks[track]
    # plt.show()
    return (tracks, old_tracks)
