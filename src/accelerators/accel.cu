#include "accel.cuh"

QUALIFIER_DEVICE Intersection Accel::TraceRay(const Ray &ray, Bsdf **bsdf_buffer,
                                              Texture **texture_buffer, const float *pixel_buffer,
                                              uint32_t *seed) const

{
    uint32_t node_stack[128];
    node_stack[0] = 0;
    int ptr = 0;
    const BvhNode *node_current = nullptr;
    Intersection its;
    // 基于栈的 BVH 先序遍历
    while (ptr >= 0)
    {
        node_current = bvh_node_buffer_ + node_stack[ptr];
        --ptr;
        while (node_current->aabb.Intersect(ray))
        {
            if (node_current->leaf)
            {
                primitive_buffer_[node_current->object_id].Intersect(ray, bsdf_buffer,
                                                                     texture_buffer,
                                                                     pixel_buffer, seed, &its);
                break;
            }
            else
            {
                ++ptr;
                node_stack[ptr] = node_current->id_right;
                node_current = bvh_node_buffer_ + node_current->id_left;
            }
        }
    }
    return its;
}