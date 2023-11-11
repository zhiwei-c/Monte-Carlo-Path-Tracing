#pragma once

#include <assimp/scene.h>
#include <string>
#include <vector>

#include "ray_tracer.cuh"

namespace rt
{

namespace model_loader
{

    Instance::Info::Meshes Load(const std::string &filename, const bool flip_texcoords,
                                const bool face_normals);
    Instance::Info::Meshes Load(const std::string &filename, const int index_shape,
                                const bool flip_texcoords, const bool face_normals);

} // namespace model_loader

} // namespace rt