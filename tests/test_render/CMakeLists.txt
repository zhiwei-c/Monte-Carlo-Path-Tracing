set(HEADER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../test_rt/model_loader.cuh")
set(SOURCE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../test_rt/model_loader.cu" "${CMAKE_CURRENT_SOURCE_DIR}/main.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${HEADER_LIST} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()


# assimp
find_package(assimp CONFIG REQUIRED)
# zlib
find_package(ZLIB REQUIRED)

add_executable(test_render ${SOURCE_LIST} ${HEADER_LIST})

target_link_libraries(test_render
    PRIVATE
        Tensor RTCore Renderer assimp::assimp ZLIB::ZLIB)
