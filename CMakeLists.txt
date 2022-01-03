# Projects Settings
cmake_minimum_required(VERSION 3.10)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

set(PROJECT_NAME "SimpleRenderer")

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
 
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/wd"6993")
endif()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


FILE(GLOB_RECURSE FILE_BSDF "${CMAKE_CURRENT_SOURCE_DIR}/src/material/*.*")
FILE(GLOB_RECURSE FILE_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/src/modeling/*.*")
FILE(GLOB FILE_RENDER "${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/*.*")
FILE(GLOB FILE_RAY_TRACING "${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/ray_tracing/*.*")
FILE(GLOB_RECURSE FILE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.*")

set(
    source_list

    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"

    ${FILE_BSDF}
    ${FILE_MODEL}
    ${FILE_RENDER}
    ${FILE_RAY_TRACING}
    ${FILE_UTILS}
)


add_executable(${PROJECT_NAME} ${source_list})

source_group("Files\\material" FILES ${FILE_BSDF})
source_group("Files\\modeling" FILES ${FILE_MODEL})
source_group("Files\\rendering" FILES ${FILE_RENDER})
source_group("Files\\rendering\\integrator" FILES ${FILE_RAY_TRACING})
source_group("Files\\utils" FILES ${FILE_UTILS})
source_group("Files" FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

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

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})