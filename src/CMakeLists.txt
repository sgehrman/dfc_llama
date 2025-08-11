# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(dfc_llama_library VERSION 0.0.1 LANGUAGES C)

add_library(dfc_llama SHARED
  "dfc_llama.c"
)

set_target_properties(dfc_llama PROPERTIES
  PUBLIC_HEADER dfc_llama.h
  OUTPUT_NAME "dfc_llama"
)

target_compile_definitions(dfc_llama PUBLIC DART_SHARED_LIB)

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(dfc_llama PRIVATE "-Wl,-z,max-page-size=16384")
endif()
