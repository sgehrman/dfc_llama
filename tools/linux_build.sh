#!/bin/bash

cd linux_build

rm -rf build

cmake -B build  
cmake --build build --config Release -j $(nproc)

rm -rf ../linux/libs/
mkdir -p ../linux/libs/

# you need -rP to preserve the symlinks
cp -rP build/bin/*.so ../linux/libs/
