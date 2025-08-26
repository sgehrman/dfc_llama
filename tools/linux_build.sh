#!/bin/bash

cd linux_build

cmake -B build  
cmake --build build --config Release -j $(nproc)
