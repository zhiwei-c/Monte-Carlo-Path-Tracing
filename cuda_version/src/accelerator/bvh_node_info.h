#pragma once

#include <algorithm>

#include "../core/shape.h"

struct BvhNodeInfo
{
    bool valid;
    bool leaf;
    uint idx;
    uint left_idx;
    uint right_idx;
    AABB aabb;
    uint obj_idx;
    Float area;

    __host__ __device__ BvhNodeInfo()
        : valid(false),
          leaf(true),
          idx(kUintMax),
          left_idx(kUintMax),
          right_idx(kUintMax),
          aabb(AABB()),
          obj_idx(kUintMax),
          area(0) {}

    BvhNodeInfo(bool leaf,
                uint idx,
                uint left_idx,
                uint right_idx,
                const AABB &aabb,
                uint obj_idx,
                Float area)
        : valid(true),
          leaf(leaf),
          idx(idx),
          left_idx(left_idx),
          right_idx(right_idx),
          aabb(aabb),
          obj_idx(obj_idx),
          area(area) {}
};

inline uint BvhNodeNum(uint num)
{

    auto height = static_cast<uint>(log2(num)) + 1;
    auto last_layer_node_num = static_cast<uint>(pow(2, height - 1));
    if (last_layer_node_num < num)
        height++;
    auto sum = static_cast<uint>(pow(2, height)) - 1;
    return sum;
}

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
    auto area_now = static_cast<Float>(0);
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