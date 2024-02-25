#pragma once

#include "../hit.cuh"
#include "../instance.cuh"
#include "../ray.cuh"
#include "bvh_builder.cuh"

namespace csrt
{

class TLAS
{
public:
    QUALIFIER_D_H TLAS();
    QUALIFIER_D_H TLAS(const Instance *instances, const BvhNode *node_buffer);

    QUALIFIER_D_H Hit Intersect(Ray *ray) const;

private:
    const BvhNode *nodes_;
    const Instance *instances_;
};

} // namespace csrt