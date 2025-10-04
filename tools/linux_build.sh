#!/bin/bash

# ===========================
# Now using Docker for builds

# We need to build llama.cpp with an older linux version to ensure compatibility
# with a wide range of systems. Otherwise if I build with a newer glibc, 
# then it will not work on older systems.

# ===========================

# cd linux_build

# rm -rf build

# cmake -B build  
# cmake --build build --config Release -j $(nproc)

# rm -rf ../linux/libs/
# mkdir -p ../linux/libs/

# ===========================

docker build -t llama-cpp-build .
docker run -e UID=$(id -u) -e GID=$(id -g) --rm -v "$(pwd)/docker_build:/output" llama-cpp-build

rm -rf ./linux/libs/
mkdir -p ./linux/libs/

cp -rP ./docker_build/*.so ./linux/libs/
