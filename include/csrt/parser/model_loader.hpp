#ifndef CSRT__PARSER__MODEL_LOADER_HPP
#define CSRT__PARSER__MODEL_LOADER_HPP

#include <assimp/scene.h>
#include <string>
#include <vector>

#include "../renderer/renderer.hpp"

namespace csrt
{

namespace model_loader
{

    MeshesInfo Load(const std::string &filename, const bool flip_texcoords,
                    const bool face_normals);
    MeshesInfo Load(const std::string &filename, const int index_shape,
                    const bool flip_texcoords, const bool face_normals);

} // namespace model_loader

} // namespace csrt

#endif