#ifndef CSRT__RTCORE__PRIMITIVES_TRIANGLE_HPP
#define CSRT__RTCORE__PRIMITIVES_TRIANGLE_HPP

#include "../accel/aabb.hpp"
#include "../hit.hpp"

namespace csrt
{

class Bsdf;

struct TriangleData
{
    Vec2 texcoords[3] = {};
    Vec3 positions[3] = {};
    Vec3 normals[3] = {};
    Vec3 tangents[3] = {};
    Vec3 bitangents[3] = {};
};

QUALIFIER_D_H AABB GetAabbTriangle(const TriangleData &data);

QUALIFIER_D_H bool IntersectTriangle(const uint32_t id_primitive,
                                     const TriangleData &data, Bsdf *bsdf,
                                     uint32_t *seed, Ray *ray, Hit *hit);

QUALIFIER_D_H Hit SampleTriangle(const uint32_t id_primitive,
                                 const TriangleData &data, const float xi_0,
                                 const float xi_1);

} // namespace csrt

#endif