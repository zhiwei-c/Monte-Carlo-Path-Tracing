file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(ENABLE_CUDA)
    set_source_files_properties(${SOURCE_LIST} ${HEADER_LIST} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${SOURCE_LIST} ${HEADER_LIST} PROPERTIES LANGUAGE CXX)
endif()

file(GLOB_RECURSE SKY_HEADER CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/*.h")
file(GLOB_RECURSE SKY_SOURCE CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/*.c")
file(GLOB_RECURSE STB_HEADER CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/extern/stb/*.h")
add_library(RayTracerLib ${SOURCE_LIST} ${HEADER_LIST} ${SKY_HEADER} ${SKY_SOURCE} ${STB_HEADER})

target_include_directories(RayTracerLib
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel"
        "${PROJECT_SOURCE_DIR}/extern/stb")
target_link_libraries(RayTracerLib
    PRIVATE
        unofficial::tinyexr::tinyexr
        assimp::assimp
        ZLIB::ZLIB
        pugixml::shared pugixml::pugixml)
if(ENABLE_CUDA)
    if(ENABLE_VIEWER)
        target_link_libraries(RayTracerLib
            PRIVATE
                $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)
    endif()
    target_compile_features(RayTracerLib PUBLIC cuda_std_17)
else()
    target_compile_features(RayTracerLib PUBLIC cxx_std_17)
endif()

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})
source_group(
    TREE "${PROJECT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES ${SOURCE_LIST})
source_group(
    TREE "${PROJECT_SOURCE_DIR}/extern"
    PREFIX "Extern Files"
    FILES ${SKY_HEADER} ${SKY_SOURCE} ${STB_HEADER})