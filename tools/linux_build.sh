#!/bin/bash

cd linux

cmake -B build  
cmake --build build --config Release -j $(nproc)
