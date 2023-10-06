# Projects Settings
cmake_minimum_required(VERSION 3.18)

option(ENABLE_CUDA "Specifies whether or not enable GPU-accelerated computing." ON)

set(PROJECT_NAME "RayTracer")
project(${PROJECT_NAME} LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENABLE_CUDA)
    message(STATUS "Enable GPU-accelerated computing")
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        #《Matching CUDA arch and CUDA gencode for various NVIDIA architectures》
        # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\"")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
else()
    message(STATUS "Disable GPU-accelerated computing")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_definitions(/wd"6993" /wd"4996" /wd"4819")
        add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
        add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    endif()
endif()

FILE(GLOB_RECURSE EXTERN_FILES "${PROJECT_SOURCE_DIR}/extern/*.h*"
                                "${PROJECT_SOURCE_DIR}/extern/*.c*")
file(GLOB_RECURSE INCLUDE_FILES "${PROJECT_SOURCE_DIR}/src/*.cuh")
file(GLOB_RECURSE SOURCES_FILES "${PROJECT_SOURCE_DIR}/src/*.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${INCLUDE_FILES} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCES_FILES} PROPERTIES LANGUAGE CXX)
endif()

add_executable(${PROJECT_NAME} ${EXTERN_FILES} ${INCLUDE_FILES} ${SOURCES_FILES})
source_group("Extern Files" FILES ${EXTERN_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})
source_group("Source Files" FILES ${SOURCES_FILES})

# ArHosekSkyModel
target_include_directories(${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel")

# assimp
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)

# pugixml
find_package(pugixml CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE pugixml::shared pugixml::pugixml)

# stb
find_package(Stb REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

# tinyexr
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

# zlib
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
