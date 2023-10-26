file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/apps/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/apps/*.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()


add_executable(RayTracer ${SOURCE_LIST} ${HEADER_LIST})

target_link_libraries(RayTracer PRIVATE RTCore)