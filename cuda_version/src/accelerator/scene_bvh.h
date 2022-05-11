#pragma once

#include "shape_bvh.h"

class SceneBvh
{
public:
    __device__ SceneBvh() : bvh_root_(nullptr) {}

    __device__ void InitSceneBvh(ShapeBvh *bvh_root)
    {
        bvh_root_ = bvh_root;
    }

    __device__ bool Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const
    {
        ShapeBvh *node_stack[64] = {nullptr};
        node_stack[0] = bvh_root_;
        int ptr = 0;
        ShapeBvh* now = nullptr;
        while (ptr >= 0)
        {
            now = node_stack[ptr];
            ptr--;
            while (now->IntersectAabb(ray))
            {
                if (now->Leaf())
                {
                    now->Intersect(ray, sample, its);
                    break;
                }
                else
                {
                    ptr++;
                    node_stack[ptr] = now->right();
                    now = now->left();
                }
            }
        }
        return its.valid();
    }

private:
    ShapeBvh *bvh_root_;
};

__global__ void CreateSceneBvh(ShapeBvh *bvh_root, SceneBvh *scene_bvh)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        scene_bvh->InitSceneBvh(bvh_root);
    }
}

__global__ void CreateSceneBvhNodes(uint max_x,
                                    uint max_y,
                                    uint scenebvh_node_num,
                                    ShapeBvh *shapebvh_list,
                                    BvhNodeInfo *scenebvh_info_list,
                                    ShapeBvh *scenebvh_node_list)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y)
        return;

    auto idx = j * max_x + i;
    if (idx >= scenebvh_node_num)
        return;

    if (!scenebvh_info_list[idx].valid)
        return;

    auto node_idx = scenebvh_info_list[idx].idx;

    if (scenebvh_info_list[idx].leaf)
        scenebvh_node_list[node_idx] = shapebvh_list[scenebvh_info_list[idx].obj_idx];
    else
        scenebvh_node_list[node_idx] = ShapeBvh(scenebvh_info_list[idx].aabb,
                                                scenebvh_info_list[idx].area,
                                                scenebvh_node_list + scenebvh_info_list[idx].left_idx,
                                                scenebvh_node_list + scenebvh_info_list[idx].right_idx);
}
