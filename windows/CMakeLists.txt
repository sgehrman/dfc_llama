# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

set(mylib_dll_bundled_libraries
  "${CMAKE_CURRENT_SOURCE_DIR}/Release/windows/Release/ggml-base.dll"
  PARENT_SCOPE
)
 
# notes
# https://github.com/ggml-org/llama.cpp/discussions/3790
# https://github.com/flutter/flutter/issues/95530
# https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20a%20Library.html

