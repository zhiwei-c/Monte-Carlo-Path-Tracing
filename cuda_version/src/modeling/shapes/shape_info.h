#pragma once

#include <string>

#include "../../rendering/intersection.h"

enum ShapeType
{
    kNoneShape,
    kMeshes,
    kRectangle,
    kCube,
    kSphere,
    kDisk
};

struct ShapeInfo
{
    ShapeType type;
    bool face_normals;
    bool flip_normals;
    bool flip_tex_coords;
    uint material_idx;
    Float radius;
    vec3 center;
    std::string filename;
    gmat4 *to_world;

    ShapeInfo(const std::string &filename,
              bool face_normals,
              bool flip_normals,
              bool flip_tex_coords,
              gmat4 *to_world,
              uint material_idx)
        : type(kMeshes),
          filename(filename),
          face_normals(face_normals),
          flip_normals(flip_normals),
          flip_tex_coords(flip_tex_coords),
          to_world(to_world),
          material_idx(material_idx) {}

    ShapeInfo(vec3 center,
              Float radius,
              bool flip_normals,
              gmat4 *to_world,
              uint material_idx)
        : type(kSphere),
          center(center),
          radius(radius),
          flip_normals(flip_normals),
          to_world(to_world),
          material_idx(material_idx) {}

    ShapeInfo(ShapeType type,
              bool flip_normals,
              gmat4 *to_world,
              uint material_idx)
        : type(type),
          flip_normals(flip_normals),
          to_world(to_world),
          material_idx(material_idx) {}

    ~ShapeInfo()
    {
        if (to_world)
        {
            delete to_world;
            to_world = nullptr;
        }
    }
};
