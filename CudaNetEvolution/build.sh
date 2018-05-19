#!/bin/bash

export CC=/usr/local/cuda/bin/gcc
export CXX=/usr/local/cuda/bin/g++

rm -rf build/
mkdir build
cd build/

cmake ..
make
