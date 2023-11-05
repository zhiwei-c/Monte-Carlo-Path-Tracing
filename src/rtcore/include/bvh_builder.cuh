#pragma once

#include "types.cuh"

#include <vector>

namespace rt
{

struct BvhNode
{
    bool leaf;
    uint32_t id;
    uint32_t id_left;
    uint32_t id_right;
    uint32_t id_object;
    float area;
    AABB aabb;

    QUALIFIER_D_H BvhNode();
    QUALIFIER_D_H BvhNode(const uint32_t _id);
    QUALIFIER_D_H BvhNode(const uint32_t _id, const uint32_t _id_object, const AABB &_aabb,
                          const float _area);
};

class BvhBuilder
{
public:
    static std::vector<BvhNode> Build(const std::vector<AABB> &aabbs,
                                      const std::vector<float> &areas);

protected:
    BvhBuilder() {}

    std::vector<BvhNode> BuildLinearBvh(const std::vector<AABB> &aabbs,
                                        const std::vector<float> &areas);
    bool GenerateMorton();
    uint32_t BuildLinearBvhTopDown(const uint32_t begin, const uint32_t end);
    uint32_t FindSplit(const uint32_t first, const uint32_t last);

    std::vector<AABB> aabbs_;
    std::vector<float> areas_;
    std::vector<uint32_t> map_id_;
    std::vector<uint64_t> mortons_;
    std::vector<BvhNode> nodes_;
};

} // namespace rt