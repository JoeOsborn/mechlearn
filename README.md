# Getting started

```
git clone git@github.com:JoeOsborn/mechlearn.git
cd mechlearn

git submodule init
git submodule update

# You should have zlib-dev installed via apt-get or brew or pacman or whatever.
# You'll need cmake and some kind of C++ compiler.
# Also, maybe make a python environment if you want to.
# it should have:
# pip install pytest

cd pybind11
mkdir build
cd build
cmake -D PYBIND11_PYTHON_VERSION=2.7 ..
make pytest -j 4 # make sure the pybind tests pass!
cp -r ../include/pybind11 /usr/local/include/ # or modify fceulib's makefile to look for these in the right place.

cd ../..

cd fceulib
make bind

#Do this if you want Python to be able to find it (and don't care abouts setting a virtualenv or the like
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`\""


cp fceulib.so ../mechlearn # put the library somewhere Python can find it later

cd ../mechlearn
```

Now we're good to go! IPython does not seem to work without messing with load paths, but CPython and I think PyPy do.
Try copying the bindtest.py script from `../fceulib` over here and it should work fine loaded up in the interpreter (`import bindtest` and `bindtest.go()` if you have a ROM called `mario.nes` in this directory).


# The full pipeline

Adam's magic from chat:

```
Illustrative.fm2 is me going through 1-1 and trying to obsessively jump while touching just about everything
Standard.fm2 is me playing 1-1 normally

If you run either of those through PPU Playground with a mario rom -   to get the pieces for the rest of this all to work, you'll need a folder called "images" and one called "nametables" (or something else if you change the code)

After running it you'll have all of the scrolled nametable id's and scrolled attribute values saved as .pngs in the nametables folder (I found this was the easiest way to package them up),  a text file called "Illustrative.txt" which are all of the sprite positions frame by frame, and a pkl of "id2sprites.pkl" which contains the actual sprite images (although it's not really used for later portions).

From there you'll need to run things through "Tracking and Filtering"  which will spit out a pickle 'mario_tracks.pkl'.  This pickle is a tuple with:
element 0 - The current tracks - a dictionary of {trackname -> track dictionary}
and
element 1 - The historic tracks (tracks that weren't active at the last time step) - a list of <trackname, track dictionary> tuples
the track dictionaries are {time : (<x,y>,sprite info)}  where <x,y> is a numpy vector and sprite info is a set of all of the sprite ids (think back to 'id2sprites.pkl')

track2 is Mario up until the time that he gets a star (I believe that's the terminus of the track, at least) and represents about 4000 frames - it should be a good amount of data

From there you can move on to Segmenation.py which actually runs the mode learning portion -  While pymc3 is doing it's thing in the middle - go over to Collisions.ipynb which will find all of the relevant collisions for the causal portion.
```
