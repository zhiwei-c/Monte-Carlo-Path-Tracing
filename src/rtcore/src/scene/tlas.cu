#include "scene.cuh"

namespace rt
{

QUALIFIER_D_H TLAS::TLAS() : nodes_(nullptr), instances_(nullptr) {}

QUALIFIER_D_H TLAS::TLAS(const uint64_t offset_node, const BvhNode *node_buffer,
                         const Instance *instances)
    : nodes_(node_buffer + offset_node), instances_(instances)
{
}

QUALIFIER_D_H Hit TLAS::Intersect(Ray *ray) const
{
    uint32_t stack[65];
    stack[0] = 0;
    int ptr = 0;
    const BvhNode *node = nullptr;
    Hit hit;
    while (ptr >= 0)
    {
        node = nodes_ + stack[ptr];
        --ptr;
        while (node->aabb.Intersect(ray))
        {
            if (node->leaf)
            {
                instances_[node->id_object].Intersect(ray, &hit);
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
    return hit;
}

} // namespace rt