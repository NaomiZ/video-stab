cmake_minimum_required(VERSION 3.12)

# If you rename the directory, CMake will default PROJECT_NAME to that dir name
project(nv_stabilizer CXX)

# Source file convention: <project_name>.cpp
set(${PROJECT_NAME}_SOURCES
    ${PROJECT_NAME}.cpp
)

# Build shared library: lib<project_name>.so
add_library(${PROJECT_NAME} SHARED
    ${${PROJECT_NAME}_SOURCES}
)

# Includes:
# - current directory      → nv_stabilizer.h
# - repo root/processing-libs → processor_api.h
#
# NOTE:
#   CMAKE_SOURCE_DIR  = top-level project root (gst-plugins/)
#   CMAKE_CURRENT_SOURCE_DIR = this directory (demo-stabilizer/)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        /home/fronti/gst-plugins/processing-libs/
        /opt/nvidia/vpi3/include    
)

target_link_directories(${PROJECT_NAME}
    PRIVATE
        /opt/nvidia/vpi3/lib/aarch64-linux-gnu/
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        nvvpi
)

# Generic properties for a plugin-style shared library
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "${PROJECT_NAME}"       # libdemo_stabilizer.so
    POSITION_INDEPENDENT_CODE ON
)

# Warnings (optional but recommended)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wpedantic
        # Uncomment if you want to treat warnings as errors:
        # -Werror
    )
endif()
