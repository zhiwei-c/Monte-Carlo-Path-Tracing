#pragma once

#include "bvh.cuh"

class NormalBvhBuilder : public BvhBuilder
{
public:
    void Build(const std::vector<AABB> &aabb_buffer, std::vector<BvhNode> *bvh_node_buffer);

private:
    uint64_t BuildBvhTopDown(const uint64_t begin, const uint64_t end);
};
