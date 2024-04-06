#ifndef CSRT__RTCORE__PRIMITIVES_DISK_HPP
#define CSRT__RTCORE__PRIMITIVES_DISK_HPP

#include "../accel/aabb.hpp"
#include "../hit.hpp"

namespace csrt
{

class Bsdf;

struct DiskData
{
    Mat4 to_world = {};
};

QUALIFIER_D_H AABB GetAabbDisk(const DiskData &data);

QUALIFIER_D_H bool IntersectDisk(const uint32_t id_primitive,
                                 const DiskData &data, Bsdf *bsdf,
                                 uint32_t *seed, Ray *ray, Hit *hit);

QUALIFIER_D_H Hit SampleDisk(const uint32_t id_primitive, const DiskData &data,
                             const float xi_0, const float xi_1);

} // namespace csrt

#endif