file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_library(RTCore ${SOURCE_LIST} ${HEADER_LIST})
target_include_directories(RTCore PUBLIC "${PROJECT_SOURCE_DIR}/include")

source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${SOURCE_LIST})