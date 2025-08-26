# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

# Project-level configuration.
set(PROJECT_NAME "llama")
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(GGML_CCACHE ON)
set(GGML_NATIVE OFF)
set(GGML_VULKAN ON)
# set(GGML_CUDA ON)
# set(CMAKE_CUDA_ARCHITECTURES "86-real")
set(LLAMA_CURL OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_TESTS OFF)

set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(GGML_BLAS_DEFAULT ON)
set(GGML_OPENMP ON)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_LIBDIR lib CACHE PATH "library install dir" FORCE)
# Set the linker flags for shared libraries 
### Windows doesn't like this?
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--build-id=none")

# Invoke the build for native code shared with the other target platforms.
# This can be changed to accommodate different builds.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src/llama.cpp" "${CMAKE_CURRENT_BINARY_DIR}/shared" EXCLUDE_FROM_ALL)

set_target_properties(llama PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN"
)

target_compile_definitions(llama PUBLIC DART_SHARED_LIB)

set(llama_bundled_libraries
  $<TARGET_FILE:llama>
  PARENT_SCOPE
)
 



# set(mylib_dll_bundled_libraries
#   "${CMAKE_CURRENT_SOURCE_DIR}/dlls/mylib.dll"
#   PARENT_SCOPE
# )
