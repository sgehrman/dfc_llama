#!/bin/bash

echo '### START building macos framework'

pushd "src/llama.cpp" > /dev/null
./build-xcframework.sh 
popd > /dev/null

rm -rf macos/llama.framework

cp -r src/llama.cpp/build-macos/framework/llama.framework macos/

rm -rf src/llama.cpp/build-macos

echo '### DONE building macos framework'
