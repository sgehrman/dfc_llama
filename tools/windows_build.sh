#!/bin/bash

cd windows_build

cmake -B build  
cmake --build build --config Release -j $(nproc)
