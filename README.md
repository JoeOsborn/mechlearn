# Getting started

```
git clone git@github.com:JoeOsborn/mechlearn.git
cd mechlearn

git submodule init
git submodule update

cd pybind11
mkdir build
cd build
cmake ..
make pytest -j 4 # make sure the pybind tests pass!
cp -r ../include/pybind11 /usr/local/include/ # or modify fceulib's makefile to look for these in the right place.

cd ../..

cd ../fceulib
make bind
cp fceulib.so ../mechlearn # put the library somewhere Python can find it later

cd ../mechlearn
```

Now we're good to go! IPython does not seem to work but CPython and I think PyPy do.
Try copying the bindtest.py script from `../fceulib` over here and it should work fine loaded up in the interpreter (`import bindtest` and `bindtest.go()` if you have a ROM called `mario.nes` in this directory).
