#pragma once

#include "linear_bvh.cuh"
#include "normal_bvh.cuh"
#include "../geometry/primitive.cuh"
#include "../renderer/intersection.cuh"

class Accel
{
public:
    QUALIFIER_DEVICE Accel(Primitive *primitive_buffer, BvhNode *bvh_node_buffer)
        : primitive_buffer_(primitive_buffer), bvh_node_buffer_(bvh_node_buffer)
    {
    }

    QUALIFIER_DEVICE bool Empty() const { return bvh_node_buffer_ == nullptr; }

    QUALIFIER_DEVICE Intersection TraceRay(const Ray &ray, Bsdf **bsdf_buffer,
                                           Texture **texture_buffer, const float *pixel_buffer,
                                           uint32_t *seed) const;

private:
    Primitive *primitive_buffer_;
    const BvhNode *bvh_node_buffer_;
};