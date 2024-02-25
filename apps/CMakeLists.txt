file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} ${SOURCE_LIST} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${HEADER_LIST} ${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_executable(RayTracer ${HEADER_LIST} ${SOURCE_LIST})

target_link_libraries(RayTracer PRIVATE RayTracerLib)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})
source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "Source Files"
    FILES ${SOURCE_LIST})