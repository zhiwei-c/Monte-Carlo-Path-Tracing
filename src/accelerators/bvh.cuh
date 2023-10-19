#pragma once

#include <vector>
#include <unordered_set>
#include <algorithm>

#include "../global.cuh"
#include "aabb.cuh"

struct BvhNode
{
    bool leaf;
    uint32_t id;
    uint32_t id_left;
    uint32_t id_right;
    uint32_t object_id;
    AABB aabb;

    QUALIFIER_DEVICE BvhNode();
    QUALIFIER_DEVICE BvhNode(const uint32_t _id, const AABB &_aabb);
    QUALIFIER_DEVICE BvhNode(const uint32_t _id, const uint32_t _object_id, const AABB &_aabb);
};

class BvhBuilder
{
public:
    enum Type
    {
        kNormal,
        kLinear,
    };

protected:
    AABB GetAabbBottomUpIndexed(const uint32_t begin, const uint32_t end);

    uint32_t max_depth_;
    uint32_t num_object_;
    AABB *aabb_buffer_;
    std::vector<uint32_t> id_map_;
    std::vector<BvhNode> *bvh_node_buffer_;
};