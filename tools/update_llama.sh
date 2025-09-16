#!/bin/bash

rm -r src
mkdir src

cd src

rm -rf ./llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

cd ..
 
dart run ffigen