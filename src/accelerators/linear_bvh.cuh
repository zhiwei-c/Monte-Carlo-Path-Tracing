#pragma once

#include "bvh.cuh"

class LinearBvhBuilder : public BvhBuilder
{
public:
    void Build(uint32_t num_object, AABB *aabb_buffer, std::vector<BvhNode> *bvh_node_buffer);

private:
    bool GenerateMorton();
    uint32_t GetMorton3D(const Vec3 &pos);
    uint32_t ExpandBits(uint32_t v);

    uint32_t BuildBvhTopDown(const uint32_t begin, const uint32_t end, const uint32_t depth);

    uint32_t FindSplit(const uint32_t first, const uint32_t last);
    int GetConsecutiveHighOrderZeroBitsNum(const uint64_t n);

    std::vector<uint64_t> morton_buffer_;
};