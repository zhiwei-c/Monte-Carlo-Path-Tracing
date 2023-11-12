set(SOURCE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/main.cu")
if(NOT ENABLE_CUDA)
    set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)
endif()

add_executable(test_render ${SOURCE_LIST})

target_link_libraries(test_render
    PRIVATE
        Tensor RTCore Renderer Parser)
