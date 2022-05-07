#pragma once

#include "shape_bvh.h"

inline void MergeShapesInfo(uint begin,
                            uint end,
                            const std::vector<uint> &shape_idx_list,
                            const std::vector<BvhNodeInfo> &shape_info_list,
                            AABB &aabb,
                            Float &area)
{
    if (begin + 1 == end)
    {
        aabb += shape_info_list[shape_idx_list[begin]].aabb;
        area += shape_info_list[shape_idx_list[begin]].area;
        return;
    }
    else if (begin + 1 > end)
        return;
    else
    {
        auto mid = (begin + end) / 2;
        MergeShapesInfo(begin, mid, shape_idx_list, shape_info_list, aabb, area);
        MergeShapesInfo(mid, end, shape_idx_list, shape_info_list, aabb, area);
    }
}

inline void BuildSceneBvhInfo(uint bvhnode_idx,
                              uint begin,
                              uint end,
                              std::vector<uint> &shape_idx_list,
                              const std::vector<BvhNodeInfo> &shape_info_list,
                              std::vector<BvhNodeInfo> &bvhnode_info_list)
{
    if (begin + 1 > end)
        return;

    if (begin + 1 == end)
    {
        bvhnode_info_list[bvhnode_idx] = BvhNodeInfo(true,
                                                     bvhnode_idx,
                                                     kUintMax,
                                                     kUintMax,
                                                     shape_info_list[begin].aabb,
                                                     shape_idx_list[begin],
                                                     shape_info_list[begin].area);
        return;
    }

    auto aabb_now = AABB();
    auto area_now = static_cast<Float>(0);
    MergeShapesInfo(begin, end, shape_idx_list, shape_info_list, aabb_now, area_now);

    auto length_x = aabb_now.max().x - aabb_now.min().x;
    auto length_y = aabb_now.max().y - aabb_now.min().y;
    auto length_z = aabb_now.max().z - aabb_now.min().z;
    if (length_x > length_y && length_x > length_z)
        std::sort(shape_idx_list.begin() + begin, shape_idx_list.begin() + end, [&](auto idx1, auto idx2)
                  { return shape_info_list[idx1].aabb.center().x < shape_info_list[idx2].aabb.center().x; });
    else if (length_y > length_z)
        std::sort(shape_idx_list.begin() + begin, shape_idx_list.begin() + end, [&](auto idx1, auto idx2)
                  { return shape_info_list[idx1].aabb.center().y < shape_info_list[idx2].aabb.center().y; });
    else
        std::sort(shape_idx_list.begin() + begin, shape_idx_list.begin() + end, [&](auto idx1, auto idx2)
                  { return shape_info_list[idx1].aabb.center().z < shape_info_list[idx2].aabb.center().z; });

    auto mid = (begin + end) / 2;
    auto left_bvh_node_idx = (bvhnode_idx + 1) * 2 - 1,
         right_bvh_node_idx = (bvhnode_idx + 1) * 2;

    BuildSceneBvhInfo(left_bvh_node_idx,
                      begin,
                      mid,
                      shape_idx_list,
                      shape_info_list,
                      bvhnode_info_list);
    BuildSceneBvhInfo(right_bvh_node_idx,
                      mid,
                      end,
                      shape_idx_list,
                      shape_info_list,
                      bvhnode_info_list);

    bvhnode_info_list[bvhnode_idx] = BvhNodeInfo(false,
                                                 bvhnode_idx,
                                                 left_bvh_node_idx,
                                                 right_bvh_node_idx,
                                                 aabb_now,
                                                 kUintMax,
                                                 area_now);
}

inline uint BvhNodeNum(uint num)
{

    auto height = static_cast<uint>(log2(num)) + 1;
    auto last_layer_node_num = static_cast<uint>(pow(2, height - 1));
    if (last_layer_node_num < num)
        height++;
    auto sum = static_cast<uint>(pow(2, height)) - 1;
    return sum;
}

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
        auto ptr = static_cast<int>(0);
        auto now = static_cast<ShapeBvh *>(nullptr);
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
