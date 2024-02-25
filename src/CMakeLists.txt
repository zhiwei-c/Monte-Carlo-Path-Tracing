file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.cuh")
file(GLOB_RECURSE SOURCE_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
if(ENABLE_CUDA)
    set_source_files_properties(${SOURCE_LIST} ${HEADER_LIST} PROPERTIES LANGUAGE CUDA)
else()
    set_source_files_properties(${SOURCE_LIST} ${HEADER_LIST} PROPERTIES LANGUAGE CXX)
endif()

set(SKY_LIST
    "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/ArHosekSkyModel.c"
    "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/ArHosekSkyModel.h"
    "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/ArHosekSkyModelData_CIEXYZ.h"
    "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/ArHosekSkyModelData_RGB.h"
    "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel/ArHosekSkyModelData_Spectral.h")
set(STB_LIST
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image.h"
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image_resize2.h"
    "${PROJECT_SOURCE_DIR}/extern/stb/stb_image_write.h")
add_library(RayTracerLib ${SOURCE_LIST} ${HEADER_LIST} ${SKY_LIST} ${STB_LIST})

target_include_directories(RayTracerLib PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(RayTracerLib PRIVATE "${PROJECT_SOURCE_DIR}/extern/ArHosekSkyModel")
target_include_directories(RayTracerLib PRIVATE "${PROJECT_SOURCE_DIR}/extern/stb")
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
    FILES ${SKY_LIST} ${STB_LIST})