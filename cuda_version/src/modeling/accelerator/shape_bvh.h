#pragma once

#include "bvh_node.h"

#include <algorithm>

inline void MergeMeshesInfo(uint begin,
                            uint end,
                            const std::vector<uint> &scene_mesh_idx_list,
                            const std::vector<AABB> &scene_mesh_aabb_list,
                            const std::vector<Float> &scene_mesh_area_list,
                            AABB &aabb,
                            Float &area)
{
    if (begin + 1 == end)
    {
        aabb += scene_mesh_aabb_list[scene_mesh_idx_list[begin]];
        area += scene_mesh_area_list[scene_mesh_idx_list[begin]];
        return;
    }
    else if (begin + 1 > end)
        return;
    else
    {
        auto mid = (begin + end) / 2;
        MergeMeshesInfo(begin,
                        mid,
                        scene_mesh_idx_list,
                        scene_mesh_aabb_list,
                        scene_mesh_area_list,
                        aabb,
                        area);
        MergeMeshesInfo(mid,
                        end,
                        scene_mesh_idx_list,
                        scene_mesh_aabb_list,
                        scene_mesh_area_list,
                        aabb,
                        area);
    }
}

inline void BuildShapeBvhInfo(uint bvhnode_idx,
                              uint begin,
                              uint end,
                              std::vector<uint> &scene_mesh_idx_list,
                              const std::vector<AABB> &scene_mesh_aabb_list,
                              const std::vector<Float> &scene_mesh_area_list,
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
                                                     scene_mesh_aabb_list[scene_mesh_idx_list[begin]],
                                                     scene_mesh_idx_list[begin],
                                                     scene_mesh_area_list[scene_mesh_idx_list[begin]]);
        return;
    }

    auto aabb_now = AABB();
    Float area_now = 0;
    MergeMeshesInfo(begin,
                    end,
                    scene_mesh_idx_list,
                    scene_mesh_aabb_list,
                    scene_mesh_area_list,
                    aabb_now,
                    area_now);

    auto length_x = aabb_now.max().x - aabb_now.min().x;
    auto length_y = aabb_now.max().y - aabb_now.min().y;
    auto length_z = aabb_now.max().z - aabb_now.min().z;
    if (length_x > length_y && length_x > length_z)
        std::sort(scene_mesh_idx_list.begin() + begin,
                  scene_mesh_idx_list.begin() + end,
                  [&](auto idx1, auto idx2)
                  { return scene_mesh_aabb_list[idx1].center().x < scene_mesh_aabb_list[idx2].center().x; });
    else if (length_y > length_z)
        std::sort(scene_mesh_idx_list.begin() + begin,
                  scene_mesh_idx_list.begin() + end,
                  [&](auto idx1, auto idx2)
                  { return scene_mesh_aabb_list[idx1].center().y < scene_mesh_aabb_list[idx2].center().y; });
    else
        std::sort(scene_mesh_idx_list.begin() + begin,
                  scene_mesh_idx_list.begin() + end,
                  [&](auto idx1, auto idx2)
                  { return scene_mesh_aabb_list[idx1].center().z < scene_mesh_aabb_list[idx2].center().z; });

    auto mid = (begin + end) / 2;
    auto left_bvhnode_idx = (bvhnode_idx + 1) * 2 - 1,
         right_bvhmode_idx = (bvhnode_idx + 1) * 2;

    BuildShapeBvhInfo(left_bvhnode_idx,
                      begin,
                      mid,
                      scene_mesh_idx_list,
                      scene_mesh_aabb_list,
                      scene_mesh_area_list,
                      bvhnode_info_list);
    BuildShapeBvhInfo(right_bvhmode_idx,
                      mid,
                      end,
                      scene_mesh_idx_list,
                      scene_mesh_aabb_list,
                      scene_mesh_area_list,
                      bvhnode_info_list);

    bvhnode_info_list[bvhnode_idx] = BvhNodeInfo(false,
                                                 bvhnode_idx,
                                                 left_bvhnode_idx,
                                                 right_bvhmode_idx,
                                                 aabb_now,
                                                 kUintMax,
                                                 area_now);
}

class ShapeBvh
{
public:
    __device__ ShapeBvh()
        : leaf_(true),
          aabb_(AABB()),
          shape_idx_(kUintMax),
          bvh_root_(nullptr),
          area_(0),
          left_(nullptr),
          right_(nullptr),
          pre_(nullptr),
          next_(nullptr) {}

    __device__ ShapeBvh(const AABB &aabb, Float area, ShapeBvh *left, ShapeBvh *right)
        : leaf_(false),
          aabb_(aabb),
          shape_idx_(kUintMax),
          bvh_root_(nullptr),
          area_(area),
          left_(left),
          right_(right),
          pre_(nullptr),
          next_(nullptr) {}

    __device__ void InitShapeBvh(uint shape_idx, BvhNode *bvh_root, Float area, ShapeBvh *pre, ShapeBvh *next)
    {
        leaf_ = true;
        shape_idx_ = shape_idx;
        bvh_root_ = bvh_root;
        aabb_ = bvh_root->aabb();
        area_ = area;
        left_ = nullptr;
        right_ = nullptr;
        pre_ = pre;
        next_ = next;
    }

    __device__ bool Leaf()
    {
        return leaf_;
    }

    __device__ bool IntersectAabb(const Ray &ray)
    {
        return aabb_.Intersect(ray);
    }

    __device__ void Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const;

    __device__ void SampleP(Intersection &its, const vec3 &sample) const;

    __device__ const AABB &aabb() const { return aabb_; }

    __device__ ShapeBvh *left() const
    {
        return left_;
    }

    __device__ ShapeBvh *right() const
    {
        return right_;
    }

private:
    bool leaf_;
    uint shape_idx_;
    AABB aabb_;
    Float area_;
    BvhNode *bvh_root_;
    ShapeBvh *left_;
    ShapeBvh *right_;
    ShapeBvh *pre_;
    ShapeBvh *next_;
};

__device__ void ShapeBvh::Intersect(const Ray &ray, const vec2 &sample, Intersection &its) const
{
    BvhNode *node_stack[64] = {nullptr};
    node_stack[0] = bvh_root_;
    auto ptr = static_cast<int>(0);
    auto now = static_cast<BvhNode *>(nullptr);
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
}

__device__ void ShapeBvh::SampleP(Intersection &its, const vec3 &sample) const
{
    auto now = bvh_root_;
    while (!now->Leaf())
    {
        if (sample.x * now->area() < now->left()->area())
            now = now->left();
        else
            now = now->right();
    }
    now->SampleP(its, sample);
}

__global__ void CreateShapeBvh(uint shapebvh_idx,
                               uint shapebvh_num,
                               BvhNode *bvh_root,
                               Float area,
                               ShapeBvh *shapebvh_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto pre = static_cast<ShapeBvh *>(nullptr);
        if (shapebvh_idx > 0)
            pre = shapebvh_list + shapebvh_idx - 1;

        auto next = static_cast<ShapeBvh *>(nullptr);
        if (shapebvh_idx + 1 < shapebvh_num)
            next = shapebvh_list + shapebvh_idx + 1;

        shapebvh_list[shapebvh_idx].InitShapeBvh(shapebvh_idx, bvh_root, area, pre, next);
    }
}

__global__ void CreateShapeBvhNodes(uint max_x,
                                    uint max_y,
                                    uint bvhnode_num,
                                    Mesh *mesh_list,
                                    BvhNodeInfo *bvhnode_info_list,
                                    BvhNode *bvhnode_list)
{
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= max_x || j >= max_y)
        return;

    auto idx = j * max_x + i;
    if (idx >= bvhnode_num)
        return;

    if (!bvhnode_info_list[idx].valid)
        return;

    auto node_idx = bvhnode_info_list[idx].idx;

    if (bvhnode_info_list[idx].leaf)
        bvhnode_list[node_idx].InitBvhNode(bvhnode_info_list[idx].aabb,
                                           mesh_list + bvhnode_info_list[idx].obj_idx,
                                           bvhnode_info_list[idx].area);
    else
        bvhnode_list[node_idx].InitBvhNode(bvhnode_info_list[idx].aabb,
                                           bvhnode_info_list[idx].area,
                                           bvhnode_list + bvhnode_info_list[idx].left_idx,
                                           bvhnode_list + bvhnode_info_list[idx].right_idx);
}
