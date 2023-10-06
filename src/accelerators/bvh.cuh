#pragma once

#include <vector>
#include <unordered_set>
#include <algorithm>

#include "../global.cuh"
#include "aabb.cuh"

struct BvhNode
{
    bool leaf;
    uint64_t id;
    uint64_t id_left;
    uint64_t id_right;
    uint64_t object_id;
    AABB aabb;

    QUALIFIER_DEVICE BvhNode();
    QUALIFIER_DEVICE BvhNode(const uint64_t _id, const AABB &_aabb);
    QUALIFIER_DEVICE BvhNode(const uint64_t _id, const uint64_t _object_id, const AABB &_aabb);
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
    AABB GetAabbBottomUpIndexed(const uint64_t begin, const uint64_t end);

    uint64_t max_depth_;
    uint64_t num_objects_;
    std::vector<AABB> aabb_buffer_;
    std::vector<uint64_t> id_map_;
    std::vector<BvhNode> *bvh_node_buffer_;
};