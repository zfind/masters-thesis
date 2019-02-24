#!/bin/bash

BUILD_DIR=build

export CC=/usr/bin/gcc-6
export CXX=/usr/bin/g++-6


#rm -rf build/
if [ ! -d "$BUILD_DIR" ]; then
    mkdir build
fi
cd $BUILD_DIR/


cmake ..
make

cp BoolGPSymbReg ../

# run
# cd ../ && ./BoolGPSymbReg parameters.txt
