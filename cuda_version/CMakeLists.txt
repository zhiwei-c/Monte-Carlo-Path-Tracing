# Projects Settings
cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "CUDA_Version")

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    #《Matching CUDA arch and CUDA gencode for various NVIDIA architectures》
    # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/wd 4819\"")

project(${PROJECT_NAME} LANGUAGES CUDA CXX)

file(GLOB ACCELERATOR_FILE "${PROJECT_SOURCE_DIR}/src/accelerator/*.*")
file(GLOB BSDFS_FILE "${PROJECT_SOURCE_DIR}/src/bsdfs/*.*")
file(GLOB EMITTERS_FILE "${PROJECT_SOURCE_DIR}/src/emitters/*.*")
file(GLOB SHAPES_FILE "${PROJECT_SOURCE_DIR}/src/shapes/*.*")
file(GLOB CORE_FILE "${PROJECT_SOURCE_DIR}/src/core/*.*")
file(GLOB CONFIG_PARSER_FILE "${PROJECT_SOURCE_DIR}/src/utils/config_parser/*.*")
file(GLOB MATH_FILE "${PROJECT_SOURCE_DIR}/src/utils/math/*.*")
file(GLOB MODEL_LOADER_FILE "${PROJECT_SOURCE_DIR}/src/utils/model_loader/*.*")
file(GLOB UTILS_FILE "${PROJECT_SOURCE_DIR}/src/utils/*.*")

# add_library(cuda_files STATIC "${PROJECT_SOURCE_DIR}/src/renderer.h"
#                                 "${PROJECT_SOURCE_DIR}/src/renderer.cu"  
#                                 ${ACCELERATOR_FILE}
#                                 ${BSDFS_FILE}
#                                 ${EMITTERS_FILE}
#                                 ${SHAPES_FILE}
#                                 ${CORE_FILE}
#                                 ${CONFIG_PARSER_FILE}
#                                 ${MATH_FILE}
#                                 ${MODEL_LOADER_FILE}
#                                 ${UTILS_FILE})

# target_compile_features(cuda_files PUBLIC cxx_std_17)
# set_target_properties(cuda_files PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cu")
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(${PROJECT_NAME} PRIVATE cuda_files)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cu"
                                "${PROJECT_SOURCE_DIR}/src/renderer.h"
                                "${PROJECT_SOURCE_DIR}/src/renderer.cu"  
                                ${ACCELERATOR_FILE}
                                ${BSDFS_FILE}
                                ${EMITTERS_FILE}
                                ${SHAPES_FILE}
                                ${CORE_FILE}
                                ${CONFIG_PARSER_FILE}
                                ${MATH_FILE}
                                ${MODEL_LOADER_FILE}
                                ${UTILS_FILE})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

source_group("Files\\accelerator" FILES ${ACCELERATOR_FILE})
source_group("Files\\bsdfs" FILES ${BSDFS_FILE})
source_group("Files\\emitters" FILES ${EMITTERS_FILE})
source_group("Files\\shapes" FILES ${SHAPES_FILE})
source_group("Files\\core" FILES ${CORE_FILE})
source_group("Files\\utils\\config_parser" FILES ${CONFIG_PARSER_FILE})
source_group("Files\\utils\\math" FILES ${MATH_FILE})
source_group("Files\\utils\\model_loader" FILES ${MODEL_LOADER_FILE})
source_group("Files\\utils" FILES ${UTILS_FILE})
source_group("Files" FILES "${PROJECT_SOURCE_DIR}/src/main.cu" 
                            "${PROJECT_SOURCE_DIR}/src/renderer.h" 
                            "${PROJECT_SOURCE_DIR}/src/renderer.cu")

# assimp
find_package(assimp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME}  PRIVATE assimp::assimp)

# RapidXML
find_path(RAPIDXML_INCLUDE_DIRS "rapidxml/rapidxml.hpp")
target_include_directories(${PROJECT_NAME} PRIVATE ${RAPIDXML_INCLUDE_DIRS})

# glm
add_compile_definitions(GLM_FORCE_CUDA CUDA_VERSION=11060)
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

# stb
find_path(STB_INCLUDE_DIRS "stb.h")
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

# tinyexr
find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE unofficial::tinyexr::tinyexr)

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})