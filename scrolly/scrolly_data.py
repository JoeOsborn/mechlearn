import retro
import itertools
from PIL import Image
import os
import csv

# read playfile
fm2file = "lordtom_tompa-smb3-100.fm2"
os.chdir("../plays")
movie = retro.Movie(fm2file)
romname = "SuperMarioBros3-Nes"
# TODO doesn't work for fm2 files, hack around it for now
print(movie.get_game())
os.chdir("../roms")
env = retro.make(game=romname, state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
os.chdir("../scrolly")
env.initial_state = movie.get_state()
env.reset()

os.makedirs("../data/{}/{}".format(romname, fm2file), exist_ok=True)

tmax = 3000
t = 0
scrolls = []
while movie.step():
    t += 1
    if tmax > 0 and t > tmax:
        break
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    print("Upto", t)
    obs, _rew, _done, info = env.step(keys)
    Image.fromarray(obs).save("../data/{}/{}/t_{}.png".format(romname, fm2file, t))
    scrolls.append((t, info["sx"], info["sy"]))

with open("../data/{}/{}/scrolls.csv".format(romname, fm2file), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(scrolls)
