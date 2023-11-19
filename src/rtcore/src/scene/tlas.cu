#include "scene.cuh"

namespace csrt
{

QUALIFIER_D_H TLAS::TLAS() : instances_(nullptr), nodes_(nullptr) {}

QUALIFIER_D_H TLAS::TLAS(const Instance *instances, const BvhNode *nodes)
    : instances_(instances), nodes_(nodes)
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

} // namespace csrt