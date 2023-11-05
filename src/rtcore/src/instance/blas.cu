#include "instance.cuh"

namespace rt
{

QUALIFIER_D_H BLAS::BLAS() : nodes_(nullptr), primitives_(nullptr) {}

QUALIFIER_D_H BLAS::BLAS(BvhNode *nodes, Primitive *primitives)
    : nodes_(nodes), primitives_(primitives)
{
}

QUALIFIER_D_H void BLAS::Intersect(Ray *ray, Hit *hit) const
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

QUALIFIER_D_H Hit rt::BLAS::Sample(const float xi_0, const float xi_1, const float xi_2) const
{
    BvhNode *node = nodes_;
    float thresh = node->area * xi_0;
    while (!node->leaf)
    {
        if (thresh < nodes_[node->id_left].area)
        {
            node = nodes_ + node->id_left;
        }
        else
        {
            node = nodes_ + node->id_right;
            thresh -= nodes_[node->id_left].area;
        }
    }

    return primitives_[node->id_object].Sample(xi_0, xi_1, xi_2);
}

} // namespace rt