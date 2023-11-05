# Projects Settings
cmake_minimum_required(VERSION 3.17)

option(ENABLE_CUDA "Specifies whether or not enable GPU-accelerated computing." ON)
option(ENABLE_WATERTIGHT_TRIANGLES "Specifies whether or not enable Woop's watertight ray/triangle intersection algorithm." ON)

project(RayTracer LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(ENABLE_CUDA)
    message(STATUS "Enable GPU-accelerated computing.")
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        #《Matching CUDA arch and CUDA gencode for various NVIDIA architectures》
        # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\"")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
else()
    message(STATUS "Disable GPU-accelerated computing")
    add_definitions(/wd"4819")
endif()

if(ENABLE_WATERTIGHT_TRIANGLES)
    message(STATUS "Enable Woop's watertight ray-triangle intersection algorithm.")
    add_definitions(-DWATERTIGHT_TRIANGLES)
else()
    message(STATUS "Enable Möller-Trumbore ray-triangle intersection algorithm.")
endif()

find_package(tinyexr CONFIG REQUIRED)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
add_subdirectory(tests)
endif()
