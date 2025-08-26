#!/bin/bash

cd windows

cmake -B build  
cmake --build build --config Release -j $(nproc)
