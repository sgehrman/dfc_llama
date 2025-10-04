#!/bin/bash

cd windows_build

rm -rf build

cmake -B build  
cmake --build build --config Release -j $(nproc)

rm -rf ../windows/dlls/
mkdir ../windows/dlls/

# you need -rP to preserve the symlinks
cp -rP build/bin/Release/*.dll ../windows/dlls/
