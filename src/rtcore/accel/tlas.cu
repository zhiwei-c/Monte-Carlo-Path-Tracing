#include "csrt/rtcore/accel/tlas.cuh"

namespace csrt
{

QUALIFIER_D_H TLAS::TLAS() : instances_(nullptr), nodes_(nullptr) {}

QUALIFIER_D_H TLAS::TLAS(const Instance *instances, const BvhNode *nodes)
    : instances_(instances), nodes_(nodes)
{
}

QUALIFIER_D_H Hit TLAS::Intersect(Bsdf *bsdf_buffer,
                                  uint32_t *map_instance_bsdf, uint32_t *seed,
                                  Ray *ray) const
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
                instances_[node->id_object].Intersect(
                    bsdf_buffer, map_instance_bsdf, seed, ray, &hit);
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
QUALIFIER_D_H bool TLAS::IntersectAny(Bsdf *bsdf_buffer,
                                      uint32_t *map_instance_bsdf,
                                      uint32_t *seed, Ray *ray) const
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
                if (instances_[node->id_object].IntersectAny(
                        bsdf_buffer, map_instance_bsdf, seed, ray))
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

} // namespace csrt