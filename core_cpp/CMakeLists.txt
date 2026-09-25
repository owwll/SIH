cmake_minimum_required(VERSION 3.22)
project(navicore_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Core Library Sources
set(NAVICORE_SOURCES
    src/fusion/eskf.cpp
    src/calibration/auto_calib.cpp
    src/zupt/nhc_zupt.cpp
    src/mapmatch/hmm_matcher.cpp
    src/pipeline/navicore_pipeline.cpp
)

# Build Static Core Library
add_library(navicore_core STATIC ${NAVICORE_SOURCES})
target_include_directories(navicore_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Test Executable
add_executable(test_eskf
    tests/fusion/test_eskf.cpp
)
target_link_libraries(test_eskf PRIVATE navicore_core)
