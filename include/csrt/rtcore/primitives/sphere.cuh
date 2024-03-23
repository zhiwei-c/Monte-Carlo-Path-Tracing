#pragma once

#include "../accel/aabb.cuh"
#include "../hit.cuh"

namespace csrt
{

class Bsdf;

struct SphereData
{
    float radius = 0;
    Vec3 center = {};
    Mat4 to_world = {};
};

QUALIFIER_D_H AABB GetAabbSphere(const SphereData &data);

QUALIFIER_D_H bool IntersectSphere(const uint32_t id_primitive,
                                   const SphereData &data, Bsdf *bsdf,
                                   uint32_t *seed, Ray *ray, Hit *hit);

QUALIFIER_D_H Hit SampleSphere(const uint32_t id_primitive,
                               const SphereData &data, const float xi_0,
                               const float xi_1);

} // namespace csrt