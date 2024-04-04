#ifndef CSRT__RTCORE__ACCEL_BLAS_HPP
#define CSRT__RTCORE__ACCEL_BLAS_HPP

#include "../primitives/primitive.hpp"
#include "bvh_builder.hpp"

namespace csrt
{

class BLAS
{
public:
    QUALIFIER_D_H BLAS();
    QUALIFIER_D_H BLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                       const uint64_t offset_primitive,
                       const Primitive *primitive_buffer);

    QUALIFIER_D_H void Intersect(Bsdf *bsdf, uint32_t *seed, Ray *ray,
                                 Hit *hit) const;
    QUALIFIER_D_H bool IntersectAny(Bsdf *bsdf, uint32_t *seed, Ray *ray) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

private:
    const BvhNode *nodes_;
    const Primitive *primitives_;
};

} // namespace csrt

#endif