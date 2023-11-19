file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
set(FILE_LIST ${HEADER_LIST} ${SOURCE_LIST})

if(NOT ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_executable(RayTracer ${HEADER_LIST} ${SOURCE_LIST})

target_link_libraries(RayTracer
    PRIVATE
        Tensor
        RTCore
        Renderer
        Parser)

if(ENABLE_CUDA AND ENABLE_VIEWER)
    find_package(FreeGLUT CONFIG REQUIRED)
    target_link_libraries(RayTracer
        PRIVATE
            $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)
endif()

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Header Files" FILES ${HEADER_LIST})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCE_LIST})