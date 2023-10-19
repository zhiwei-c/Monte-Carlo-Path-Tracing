#pragma once

#include "bvh.cuh"

class NormalBvhBuilder : public BvhBuilder
{
public:
    void Build(uint32_t num_object, AABB *aabb_buffer, std::vector<BvhNode> *bvh_node_buffer);

private:
    uint32_t BuildBvhTopDown(const uint32_t begin, const uint32_t end);
};
