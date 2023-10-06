#include "normal_bvh.cuh"

void NormalBvhBuilder::Build(const std::vector<AABB> &aabb_buffer,
                             std::vector<BvhNode> *bvh_node_buffer)
{
    aabb_buffer_ = aabb_buffer;
    bvh_node_buffer_ = bvh_node_buffer;

    num_objects_ = aabb_buffer.size();
    id_map_ = std::vector<uint64_t>(num_objects_);
    for (uint64_t i = 0; i < num_objects_; ++i)
        id_map_[i] = i;

    *bvh_node_buffer_ = {};
    BuildBvhTopDown(0, num_objects_);
}

uint64_t NormalBvhBuilder::BuildBvhTopDown(const uint64_t begin, const uint64_t end)
{
    const uint64_t id_node = bvh_node_buffer_->size();

    if (begin + 1 == end)
    {
        bvh_node_buffer_->push_back(BvhNode(id_node, id_map_[begin], aabb_buffer_[id_map_[begin]]));
        return id_node;
    }

    const AABB aabb_current = GetAabbBottomUpIndexed(begin, end);
    int dim_target;
    float length_max = kLowestFloat;
    for (int dim = 0; dim < 3; ++dim)
    {
        const float length_current = (aabb_current.max()[dim] -
                                      aabb_current.min()[dim]);
        if (length_current > length_max)
        {
            dim_target = dim;
            length_max = length_current;
        }
    }
    std::sort(id_map_.begin() + begin, id_map_.begin() + end,
              [&](const uint64_t id1, const uint64_t id2)
              { return aabb_buffer_[id1].center()[dim_target] <
                       aabb_buffer_[id2].center()[dim_target]; });

    bvh_node_buffer_->push_back(BvhNode(id_node, aabb_current));
    const uint64_t middle = (begin + end) / 2;
    (*bvh_node_buffer_)[id_node].id_left = BuildBvhTopDown(begin, middle);
    (*bvh_node_buffer_)[id_node].id_right = BuildBvhTopDown(middle, end);
    return id_node;
}
