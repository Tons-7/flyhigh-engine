# A C host of the FlyHigh engine against the built C library.
#   cargo build --release -p flyhigh-capi                      (from the repository root)
#   cmake -S examples/c -B build/c && cmake --build build/c    (FLYHIGH_LIB_DIR points elsewhere than target/release)
#   ./build/c/flyhigh_c_example models
cmake_minimum_required(VERSION 3.16)
project(flyhigh_c_example C)

set(FLYHIGH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(FLYHIGH_LIB_DIR "${FLYHIGH_ROOT}/target/release" CACHE PATH "directory with the built flyhigh library")
set(FLYHIGH_INCLUDE_DIR "${FLYHIGH_ROOT}/crates/flyhigh-capi/include" CACHE PATH "directory with flyhigh.h")

add_executable(flyhigh_c_example main.c)
target_include_directories(flyhigh_c_example PRIVATE "${FLYHIGH_INCLUDE_DIR}")
if(WIN32)
    # MSVC links the import library, MinGW the DLL itself
    find_library(FLYHIGH_LIBRARY NAMES flyhigh.dll.lib flyhigh HINTS "${FLYHIGH_LIB_DIR}" NO_DEFAULT_PATH)
else()
    find_library(FLYHIGH_LIBRARY NAMES flyhigh HINTS "${FLYHIGH_LIB_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT FLYHIGH_LIBRARY)
    message(FATAL_ERROR "flyhigh library not found in ${FLYHIGH_LIB_DIR}: build it with `cargo build --release -p flyhigh-capi`")
endif()
target_link_libraries(flyhigh_c_example PRIVATE "${FLYHIGH_LIBRARY}")
if(NOT WIN32)
    target_link_libraries(flyhigh_c_example PRIVATE m)
    set_target_properties(flyhigh_c_example PROPERTIES BUILD_RPATH "${FLYHIGH_LIB_DIR}")
endif()
