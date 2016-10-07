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

cd ../fceulib
make bind

#Do this if you want Python to be able to find it (and don't care abouts setting a virtualenv or the like
echo "export PYTHONPATH=\"\$PYTHONPATH:`pwd`\""


cp fceulib.so ../mechlearn # put the library somewhere Python can find it later

cd ../mechlearn
```

Now we're good to go! IPython does not seem to work but CPython and I think PyPy do.
Try copying the bindtest.py script from `../fceulib` over here and it should work fine loaded up in the interpreter (`import bindtest` and `bindtest.go()` if you have a ROM called `mario.nes` in this directory).
