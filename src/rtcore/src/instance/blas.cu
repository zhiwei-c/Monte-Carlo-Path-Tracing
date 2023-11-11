#include "instance.cuh"

namespace rt
{

QUALIFIER_D_H BLAS::BLAS() : nodes_(nullptr), primitives_(nullptr) {}

QUALIFIER_D_H BLAS::BLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                         const uint64_t offset_primitive,
                         const Primitive *primitive_buffer)
    : nodes_(node_buffer + offset_node),
      primitives_(primitive_buffer + offset_primitive)
{
}

QUALIFIER_D_H void BLAS::Intersect(Ray *ray, Hit *hit) const
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
                primitives_[node->id_object].Intersect(ray, hit);
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

QUALIFIER_D_H Hit rt::BLAS::Sample(const Vec3 &xi) const
{
    const BvhNode *node = nodes_;
    float thresh = node->area * xi.x;
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

    return primitives_[node->id_object].Sample(xi.y, xi.z);
}

} // namespace rt