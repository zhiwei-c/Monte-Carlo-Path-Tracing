# Projects Settings
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "CPU_Version")

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

add_definitions(/wd"6993" /wd"4996" /wd"4819")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

project(${PROJECT_NAME} LANGUAGES CXX)

FILE(GLOB_RECURSE FILE_ACC "${CMAKE_CURRENT_SOURCE_DIR}/src/accelerator/*.*")
FILE(GLOB_RECURSE FILE_BSDFS "${CMAKE_CURRENT_SOURCE_DIR}/src/bsdfs/*.*")
FILE(GLOB_RECURSE FILE_CORE "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.*")
FILE(GLOB_RECURSE FILE_EMITTERS "${CMAKE_CURRENT_SOURCE_DIR}/src/emitters/*.*")
FILE(GLOB_RECURSE FILE_INTEGRATORS "${CMAKE_CURRENT_SOURCE_DIR}/src/integrators/*.*")
FILE(GLOB_RECURSE FILE_MICROFACET_DISTRIB "${CMAKE_CURRENT_SOURCE_DIR}/src/microfacet_distribution/*.*")
FILE(GLOB_RECURSE FILE_SHAPES "${CMAKE_CURRENT_SOURCE_DIR}/src/shapes/*.*")
FILE(GLOB_RECURSE FILE_TEXTURE "${CMAKE_CURRENT_SOURCE_DIR}/src/textures/*.*")
FILE(GLOB FILE_CONFIG_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/config_parser/*.*")
FILE(GLOB FILE_MODEL_LOADER "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/model_loader/*.*")
FILE(GLOB FILE_MATH "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/math/*.*")
FILE(GLOB FILE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.*")

add_executable(${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.h"
                                ${FILE_ACC}
                                ${FILE_BSDFS}
                                ${FILE_CORE}
                                ${FILE_EMITTERS}
                                ${FILE_INTEGRATORS}
                                ${FILE_MICROFACET_DISTRIB}
                                ${FILE_SHAPES}
                                ${FILE_TEXTURE}
                                ${FILE_CONFIG_PARSER}
                                ${FILE_MODEL_LOADER}
                                ${FILE_MATH}
                                ${FILE_UTILS})

source_group("Files\\accelerator" FILES ${FILE_ACC})
source_group("Files\\bsdfs" FILES ${FILE_BSDFS})
source_group("Files\\core" FILES ${FILE_CORE})
source_group("Files\\emitters" FILES ${FILE_EMITTERS})
source_group("Files\\integrators" FILES ${FILE_INTEGRATORS})
source_group("Files\\microfacet distribution" FILES ${FILE_MICROFACET_DISTRIB})
source_group("Files\\shapes" FILES ${FILE_SHAPES})
source_group("Files\\textures" FILES ${FILE_TEXTURE})
source_group("Files\\utils\\config parser" FILES ${FILE_CONFIG_PARSER})
source_group("Files\\utils\\model loader" FILES ${FILE_MODEL_LOADER})
source_group("Files\\utils\\math" FILES ${FILE_MATH})
source_group("Files\\utils" FILES ${FILE_UTILS})
source_group("Files" FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
source_group("Files" FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.h")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# nlohmann json
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json nlohmann_json::nlohmann_json)

# RapidXML
find_path(RAPIDXML_INCLUDE_DIRS "rapidxml/rapidxml.hpp")
target_include_directories(${PROJECT_NAME} PRIVATE ${RAPIDXML_INCLUDE_DIRS})

# glm
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# assimp
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}  PRIVATE assimp::assimp)

# tinyobjloader
find_package(tinyobjloader CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader::tinyobjloader)

# stb
find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

# tinyexr
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})