#pragma once

#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <memory>

#include "../core/bsdf.h"
#include "../core/shape.h"

NAMESPACE_BEGIN(raytracer)

class ModelParser
{
public:
    static void Parse(const std::string &obj_path, std::vector<Shape *> &shapes, std::vector<Bsdf *> &bsdfs);

    static Meshes *Parse(std::string filename, Bsdf *bsdf, Medium *medium, std::unique_ptr<Mat4> to_world, bool flip_normals, bool face_normals, bool flip_tex_coords = false);
};

NAMESPACE_END(raytracer)