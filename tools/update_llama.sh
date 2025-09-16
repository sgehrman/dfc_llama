#!/bin/bash

rm -rf src
mkdir src

cd src

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

cd ..
 
dart run ffigen