set(HEADER_PUBLIC_LIST "${PROJECT_SOURCE_DIR}/include/rtcore.cuh")
file(GLOB_RECURSE HEADER_PRIVATE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.*")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.*")
set(FILE_LIST ${HEADER_PUBLIC_LIST} ${HEADER_PRIVATE_LIST} ${SOURCE_LIST})

if(NOT ENABLE_CUDA)
    set_source_files_properties(${FILE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_library(RTCore ${FILE_LIST})

target_link_libraries(RTCore
    PUBLIC
        Tensor
    PRIVATE
        Utils)

target_include_directories(RTCore
    INTERFACE
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_PRIVATE_LIST})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCE_LIST})