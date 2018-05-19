#!/bin/bash
set -e
set -x

export CC=/usr/local/cuda/bin/gcc
export CXX=/usr/local/cuda/bin/g++

cd "$( dirname "${BASH_SOURCE[0]}" )"

rm -rf build
mkdir build
cd build
cmake .. $@
make
