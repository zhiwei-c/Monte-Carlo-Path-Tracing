# Projects Settings
cmake_minimum_required(VERSION 3.10)

# vcpkg
set(CMAKE_TOOLCHAIN_FILE "D:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(Simple_Renderer)

add_subdirectory(${PROJECT_SOURCE_DIR}/cpu_version)
add_subdirectory(${PROJECT_SOURCE_DIR}/cuda_version)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CPU_Version)