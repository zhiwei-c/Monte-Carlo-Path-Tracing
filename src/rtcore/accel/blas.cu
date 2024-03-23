#include "csrt/rtcore/accel/blas.cuh"

#include "csrt/renderer/bsdfs/bsdf.cuh"

namespace csrt
{

QUALIFIER_D_H BLAS::BLAS() : nodes_(nullptr), primitives_(nullptr) {}

QUALIFIER_D_H BLAS::BLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                         const uint64_t offset_primitive,
                         const Primitive *primitive_buffer)
    : nodes_(node_buffer + offset_node),
      primitives_(primitive_buffer + offset_primitive)
{
}

QUALIFIER_D_H void BLAS::Intersect(Bsdf *bsdf, uint32_t *seed, Ray *ray,
                                   Hit *hit) const
{
    uint32_t stack[65];
    stack[0] = 0;
    int ptr = 0;
    const BvhNode *node = nullptr;
    while (ptr >= 0)
    {
        node = nodes_ + stack[ptr];
        --ptr;
        while (node->aabb.Intersect(ray))
        {
            if (node->leaf)
            {
                primitives_[node->id_object].Intersect(bsdf, seed, ray, hit);
                break;
            }
            else
            {
                ++ptr;
                stack[ptr] = node->id_right;
                node = nodes_ + node->id_left;
            }
        }
    }
}

QUALIFIER_D_H bool BLAS::IntersectAny(Bsdf *bsdf, uint32_t *seed,
                                      Ray *ray) const
{
    uint32_t stack[65];
    stack[0] = 0;
    int ptr = 0;
    const BvhNode *node = nullptr;
    while (ptr >= 0)
    {
        node = nodes_ + stack[ptr];
        --ptr;
        while (node->aabb.Intersect(ray))
        {
            if (node->leaf)
            {
                if (primitives_[node->id_object].Intersect(bsdf, seed, ray,
                                                           nullptr))
                    return true;
                else
                    break;
            }
            else
            {
                ++ptr;
                stack[ptr] = node->id_right;
                node = nodes_ + node->id_left;
            }
        }
    }

    return false;
}

QUALIFIER_D_H Hit BLAS::Sample(const float xi_0, const float xi_1,
                               const float xi_2) const
{
    const BvhNode *node = nodes_;
    float thresh = node->area * xi_0;
    while (!node->leaf)
    {
        if (thresh < nodes_[node->id_left].area)
        {
            node = nodes_ + node->id_left;
        }
        else
        {
            thresh -= nodes_[node->id_left].area;
            node = nodes_ + node->id_right;
        }
    }

    return primitives_[node->id_object].Sample(xi_1, xi_2);
}

} // namespace csrt