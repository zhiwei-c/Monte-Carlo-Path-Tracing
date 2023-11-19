set(HEADER_PUBLIC_LIST "${PROJECT_SOURCE_DIR}/include/utils.cuh")
file(GLOB_RECURSE HEADER_PRIVATE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.*")
set(FILE_LIST ${HEADER_PUBLIC_LIST} ${HEADER_PRIVATE_LIST} ${SOURCE_LIST})

set(STB_LIST
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image.h"
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image_resize2.h"
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image_write.h")

if(NOT ENABLE_CUDA)
    set_source_files_properties(${FILE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_library(Utils ${FILE_LIST} ${STB_LIST})

find_package(tinyexr CONFIG REQUIRED)
target_link_libraries(Utils
    PUBLIC
    Tensor
PRIVATE
    unofficial::tinyexr::tinyexr)

target_include_directories(Utils
    INTERFACE
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/extern/stb")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_PRIVATE_LIST})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCE_LIST})
source_group(TREE "${PROJECT_SOURCE_DIR}/extern" PREFIX "Extern Files" FILES ${STB_LIST})