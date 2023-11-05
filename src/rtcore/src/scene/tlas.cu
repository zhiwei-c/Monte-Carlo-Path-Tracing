#include "scene.cuh"

namespace rt
{

QUALIFIER_D_H TLAS::TLAS() : nodes_(nullptr), instances_(nullptr) {}

QUALIFIER_D_H TLAS::TLAS(BvhNode *nodes, Instance *instances) : nodes_(nodes), instances_(instances)
{
}

QUALIFIER_D_H void TLAS::Intersect(Ray *ray, Hit *hit) const
{
    uint32_t stack[65];
    stack[0] = 0;
    int ptr = 0;
    BvhNode *node = nullptr;
    while (ptr >= 0)
    {
        node = nodes_ + stack[ptr];
        --ptr;
        while (node->aabb.Intersect(ray))
        {
            if (node->leaf)
            {
                instances_[node->id_object].Intersect(ray, hit);
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

} // namespace rt