cmake_minimum_required (VERSION 3.10)

project ("KDRay")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory ("kdray")
add_subdirectory ("viewer")
add_subdirectory ("converter")