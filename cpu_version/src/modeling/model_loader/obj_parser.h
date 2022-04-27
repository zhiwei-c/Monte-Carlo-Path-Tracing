#pragma once

#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <memory>

#include "../../utils/obj_loader/obj_loader.h"
#include "../../material/bsdfs/bsdfs.h"
#include "../shapes/shapes.h"

NAMESPACE_BEGIN(simple_renderer)

class ObjParser
{
public:
    static void Parse(const std::string &obj_path, std::vector<Shape *> &shapes, std::vector<Material *> &materials);

    static Meshes *Parse(std::string filename, Material *material, std::unique_ptr<Mat4> to_world, bool flip_normals, bool face_normals, bool flip_tex_coords = false);

};

NAMESPACE_END(simple_renderer)