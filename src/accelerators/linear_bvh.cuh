#pragma once

#include "bvh.cuh"

class LinearBvhBuilder : public BvhBuilder
{
public:
    void Build(const std::vector<AABB> &aabb_buffer_, std::vector<BvhNode> *bvh_node_buffer);

private:
    bool GenerateMorton();
    uint64_t GetMorton3D(const Vec3 &pos);
    uint64_t ExpandBits(uint64_t v);

    uint64_t BuildBvhTopDown(const uint64_t begin, const uint64_t end, const uint64_t depth);

    uint64_t FindSplit(const uint64_t first, const uint64_t last);
    uint64_t GetConsecutiveHighOrderZeroBitsNum(const uint64_t n);

    std::vector<uint64_t> morton_buffer_;
};