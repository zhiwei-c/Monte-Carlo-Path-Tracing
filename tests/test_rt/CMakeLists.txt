file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()

# assimp
find_package(assimp CONFIG REQUIRED)
# zlib
find_package(ZLIB REQUIRED)

add_executable(test_rt ${SOURCE_LIST} ${HEADER_LIST})

target_link_libraries(test_rt
    PRIVATE
        Tensor RTCore Renderer assimp::assimp ZLIB::ZLIB)
