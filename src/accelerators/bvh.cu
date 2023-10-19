#include "bvh.cuh"

QUALIFIER_DEVICE BvhNode::BvhNode()
    : leaf(true), id(kInvalidId), id_left(kInvalidId), id_right(kInvalidId),
      object_id(kInvalidId), aabb(AABB())
{
}

QUALIFIER_DEVICE BvhNode::BvhNode(const uint32_t _id, const AABB &_aabb)
    : leaf(false), id(_id), id_left(kInvalidId), id_right(kInvalidId),
      object_id(kInvalidId), aabb(_aabb)
{
}

QUALIFIER_DEVICE BvhNode::BvhNode(const uint32_t _id, const uint32_t _object_id, const AABB &_aabb)
    : leaf(true), id(_id), id_left(kInvalidId), id_right(kInvalidId),
      object_id(_object_id), aabb(_aabb)
{
}

AABB BvhBuilder::GetAabbBottomUpIndexed(const uint32_t begin, const uint32_t end)
{
    if (begin + 1 == end)
    {
        return aabb_buffer_[id_map_[begin]];
    }
    else if (begin + 1 > end)
    {
        return AABB();
    }
    else
    {
        const uint32_t mid = (begin + end) / 2;
        return GetAabbBottomUpIndexed(begin, mid) + GetAabbBottomUpIndexed(mid, end);
    }
};
