* Mechlearn

This is the joint repository for CHARDA [1], MARIO [2], Mappy [3], and related projects in automated game design learning [4].

Real documentation is coming soon!  For information on how to run these programs in the meantime, please contact jcosborn@ucsc.edu.

Preliminary documentation:

* General Setup
```bash
git clone git@github.com:JoeOsborn/mechlearn.git
cd mechlearn
git submodule init
git submodule update
# You should have zlib-dev and opencv3 installed via apt-get or brew or pacman or whatever.
# You will need cmake and some kind of C++ compiler.
# Also, maybe make a python environment if you want to.
# it should have:
pip install pytest cv2wrap pillow numpy matplotlib scipy scikit-learn jupyter networkx

cd pybind11
mkdir build
cd build
cmake -D PYBIND11_PYTHON_VERSION=2.7 && make pytest -j 4

cd ../..

cd fceulib
make -j 4 bind

cp fceulib.so ../mechlearn # put the library somewhere Python can find it later

cd ../mechlearn

jupyter notebook
```

Make sure you have all your ROMs, fm2 files, etc set up, then:

Mappy: `mappy.ipynb`
MARIO: `jumpfit.ipynb`
CHARDA: `make all`

Details on that will follow once we refresh MARIO, CHARDA, and Mappy to use our new instrumenting NES emulator.

[1] https://arxiv.org/abs/1707.03336

[2] https://arxiv.org/abs/1707.03865

[3] https://arxiv.org/abs/1707.03908

[4] https://arxiv.org/abs/1707.03333
