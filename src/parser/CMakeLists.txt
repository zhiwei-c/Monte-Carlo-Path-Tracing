set(HEADER_PUBLIC_LIST "${PROJECT_SOURCE_DIR}/include/parser.cuh")
file(GLOB_RECURSE HEADER_PRIVATE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.*")
set(FILE_LIST ${HEADER_PUBLIC_LIST} ${HEADER_PRIVATE_LIST} ${SOURCE_LIST})

if(NOT ENABLE_CUDA)
    set_source_files_properties(${FILE_LIST} PROPERTIES LANGUAGE CXX)
endif()


add_library(Parser ${FILE_LIST})

find_package(assimp CONFIG REQUIRED)
find_package(pugixml CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
target_link_libraries(Parser
    PUBLIC
        Tensor
        RTCore
        Renderer
    PRIVATE
        Utils
        assimp::assimp
        ZLIB::ZLIB
        pugixml::shared pugixml::pugixml)

target_include_directories(Parser
    INTERFACE
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_PRIVATE_LIST})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCE_LIST})