#pragma once

#include "../primitives/primitive.cuh"
#include "bvh_builder.cuh"

namespace csrt
{

class BLAS
{
public:
    QUALIFIER_D_H BLAS();
    QUALIFIER_D_H BLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                       const uint64_t offset_primitive,
                       const Primitive *primitive_buffer);

    QUALIFIER_D_H void Intersect(Ray *ray, Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1,
                             const float xi_2) const;

private:
    const BvhNode *nodes_;
    const Primitive *primitives_;
};

} // namespace csrt