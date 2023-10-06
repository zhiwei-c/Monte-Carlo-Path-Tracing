#pragma once

#include "../global.hpp"
#include "../shapes/shape.hpp"

NAMESPACE_BEGIN(raytracer)

std::vector<Shape *> LoadObject(const std::string &filename, int shape_index, bool face_normals, bool flip_normals,
                                bool flip_tex_coords, const std::string &id, const dmat4 &to_world);

NAMESPACE_END(raytracer)