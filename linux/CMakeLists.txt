# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)
 
file(GLOB DLL_LIST "${CMAKE_CURRENT_SOURCE_DIR}/libs/*.so")
set(dfc_llama_bundled_libraries
  ${DLL_LIST}
  PARENT_SCOPE
)
