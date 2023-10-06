# Projects Settings
cmake_minimum_required(VERSION 3.10)

# vcpkg
# set(CMAKE_TOOLCHAIN_FILE "D:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

set(PROJECT_NAME "RayTracer")

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

project(${PROJECT_NAME} LANGUAGES CXX C)

option(ROUGH_SMOOTH "Use rough surface with small roughness instead of smooth surface" ON)
if (ROUGH_SMOOTH)
    message(STATUS "Use rough surface with small roughness instead of smooth surface.")
    add_definitions(-DROUGH_SMOOTH)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/wd"6993" /wd"4996" /wd"4819")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

FILE(GLOB FILE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.*")
FILE(GLOB FILE_ACC "${CMAKE_CURRENT_SOURCE_DIR}/src/accelerators/*.*")
FILE(GLOB FILE_BSDFS "${CMAKE_CURRENT_SOURCE_DIR}/src/bsdfs/*.*")
FILE(GLOB FILE_CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.*")
FILE(GLOB FILE_EMITTERS "${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/*.*")
FILE(GLOB FILE_INTEGRATORS "${CMAKE_CURRENT_SOURCE_DIR}/src/integrators/*.*")
FILE(GLOB FILE_MATH "${CMAKE_CURRENT_SOURCE_DIR}/src/math/*.*")
FILE(GLOB FILE_MEDIA "${CMAKE_CURRENT_SOURCE_DIR}/src/media/*.*")
FILE(GLOB FILE_PHASE "${CMAKE_CURRENT_SOURCE_DIR}/src/media/phase_functions/*.*")
FILE(GLOB FILE_NDFS "${CMAKE_CURRENT_SOURCE_DIR}/src/ndfs/*.*")
FILE(GLOB FILE_SHAPES "${CMAKE_CURRENT_SOURCE_DIR}/src/shapes/*.*")
FILE(GLOB FILE_TEXTURE "${CMAKE_CURRENT_SOURCE_DIR}/src/textures/*.*")
FILE(GLOB FILE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.*")

FILE(GLOB FILE_SKY_MODEL_H "${CMAKE_CURRENT_SOURCE_DIR}/extern/ArHosekSkyModel/*.h")
FILE(GLOB FILE_SKY_MODEL_C "${CMAKE_CURRENT_SOURCE_DIR}/extern/ArHosekSkyModel/*.c")

add_executable(${PROJECT_NAME} ${FILE_ROOT}
                                ${FILE_ACC}
                                ${FILE_BSDFS}
                                ${FILE_CORE}
                                ${FILE_SKY_MODEL}
                                ${FILE_EMITTERS}
                                ${FILE_INTEGRATORS}
                                ${FILE_MATH}
                                ${FILE_MEDIA}
                                ${FILE_PHASE}
                                ${FILE_NDFS}
                                ${FILE_SHAPES}
                                ${FILE_TEXTURE}
                                ${FILE_UTILS}

                                ${FILE_SKY_MODEL_H}
                                ${FILE_SKY_MODEL_C}
                                )

                                
source_group("Files" FILES ${FILE_ROOT})

source_group("Files\\accelerator" FILES ${FILE_ACC})
source_group("Files\\bsdfs" FILES ${FILE_BSDFS})
source_group("Files\\core" FILES ${FILE_CORE})
source_group("Files\\emitters" FILES ${FILE_EMITTERS})
source_group("Files\\integrators" FILES ${FILE_INTEGRATORS})
source_group("Files\\math" FILES ${FILE_MATH})
source_group("Files\\media" FILES ${FILE_MEDIA})
source_group("Files\\phase functions" FILES ${FILE_PHASE})
source_group("Files\\normal distribution functions" FILES ${FILE_NDFS})
source_group("Files\\shapes" FILES ${FILE_SHAPES})
source_group("Files\\texture" FILES ${FILE_TEXTURE})
source_group("Files\\utils" FILES ${FILE_UTILS})

source_group("extern\\ArHosekSkyModel" FILES ${FILE_SKY_MODEL_H} ${FILE_SKY_MODEL_C})

# zlib
find_package(ZLIB REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ZLIB::ZLIB)

# pugixml
find_package(pugixml CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE pugixml pugixml::shared pugixml::pugixml)

# glm
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# assimp
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}  PRIVATE assimp::assimp)

# stb
find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

# tinyexr
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

# ArHosekSkyModel
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/extern/ArHosekSkyModel")

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})