#pragma once

#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <memory>

#include "../core/material.h"
#include "../core/shape.h"

NAMESPACE_BEGIN(raytracer)

class ModelParser
{
public:
    static void Parse(const std::string &obj_path, std::vector<Shape *> &shapes, std::vector<Material *> &materials);

    static Meshes *Parse(std::string filename, Material *material, std::unique_ptr<Mat4> to_world, bool flip_normals, bool face_normals, bool flip_tex_coords = false);
};

NAMESPACE_END(raytracer)