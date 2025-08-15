#!/bin/bash

echo '### START building macos framework'

# must rebuild ffi, the one in git is built for linux
# dart run ffigen (does it in update_llama.sh)
./update_llama.sh

pushd "src/llama.cpp" > /dev/null
./build-xcframework.sh 
popd > /dev/null

rm -rf macos/llama.framework

# you need -rP to preserve the symlinks
cp -rP src/llama.cpp/build-macos/framework/llama.framework macos/

rm -rf src/llama.cpp/build-macos

echo '### DONE building macos framework'
