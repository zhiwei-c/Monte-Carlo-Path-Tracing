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

project(${PROJECT_NAME} LANGUAGES CUDA CXX)

file(GLOB ACCELERATOR_FILE "${PROJECT_SOURCE_DIR}/src/modeling/accelerator/*.*")
file(GLOB MATERIALS_FILE "${PROJECT_SOURCE_DIR}/src/modeling/materials/*.*")
file(GLOB SHAPES_FILE "${PROJECT_SOURCE_DIR}/src/modeling/shapes/*.*")
file(GLOB TEXTURES_FILE "${PROJECT_SOURCE_DIR}/src/modeling/textures/*.*")
file(GLOB MODELING_FILE "${PROJECT_SOURCE_DIR}/src/modeling/*.*")
file(GLOB RENDERING_FILE "${PROJECT_SOURCE_DIR}/src/rendering/*.*")
file(GLOB MATH_FILE "${PROJECT_SOURCE_DIR}/src/utils/math/*.*")
file(GLOB MODEL_LOADER_FILE "${PROJECT_SOURCE_DIR}/src/utils/model_loader/*.*")
file(GLOB UTILS_FILE "${PROJECT_SOURCE_DIR}/src/utils/*.*")

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cu"
                               "${PROJECT_SOURCE_DIR}/src/renderer.h"
                               ${ACCELERATOR_FILE}
                               ${MATERIALS_FILE}
                               ${SHAPES_FILE}
                               ${TEXTURES_FILE}
                               ${MODELING_FILE}
                               ${RENDERING_FILE}
                               ${MATH_FILE}
                               ${MODEL_LOADER_FILE}
                               ${UTILS_FILE})

source_group("Files\\modeling\\accelerator" FILES ${ACCELERATOR_FILE})
source_group("Files\\modeling\\materials" FILES ${MATERIALS_FILE})
source_group("Files\\modeling\\shapes" FILES ${SHAPES_FILE})
source_group("Files\\modeling\\textures" FILES ${TEXTURES_FILE})
source_group("Files\\modeling" FILES ${MODELING_FILE})
source_group("Files\\rendering" FILES ${RENDERING_FILE})
source_group("Files\\utils\\math" FILES ${MATH_FILE})
source_group("Files\\utils\\model_loader" FILES ${MODEL_LOADER_FILE})
source_group("Files\\utils" FILES ${UTILS_FILE})
source_group("Files" FILES "${PROJECT_SOURCE_DIR}/src/main.cu" "${PROJECT_SOURCE_DIR}/src/renderer.h")

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