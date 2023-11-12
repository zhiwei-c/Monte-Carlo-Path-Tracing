#include "bvh_builder.cuh"

#include <algorithm>
#include <exception>
#include <unordered_set>

#include "utils.cuh"

namespace
{

using namespace csrt;

// Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
uint32_t ExpandBits(uint32_t v)
{
    v = (v * ((0x1ul << 16) + 1)) & 0xFF0000FFul;
    v = (v * ((0x1ul << 8) + 1)) & 0x0F00F00Ful;
    v = (v * ((0x1ul << 4) + 1)) & 0xC30C30C3ul;
    v = (v * ((0x1ul << 2) + 1)) & 0x49249249ul;
    return v;
}

int GetConsecutiveHighOrderZeroBitsNum(const uint64_t n)
{
    int count = 0;
    for (int i = 0; i < 64; ++i)
    {
        if ((n >> (63 - i)) & 0x1)
            break;
        else
            ++count;
    }
    return count;
}

// Calculates a 30-bit Morton code for the given 3D point located within the
// unit cube [0,1].
uint32_t GetMorton3D(const Vec3 &v)
{
    const float x = fminf(fmaxf(v.x * 1024.0f, 0.0f), 1023.0f),
                y = fminf(fmaxf(v.y * 1024.0f, 0.0f), 1023.0f),
                z = fminf(fmaxf(v.z * 1024.0f, 0.0f), 1023.0f);
    const uint32_t xx = ExpandBits(static_cast<uint32_t>(x)),
                   yy = ExpandBits(static_cast<uint32_t>(y)),
                   zz = ExpandBits(static_cast<uint32_t>(z));
    return xx * 4 + yy * 2 + zz;
}

} // namespace

namespace csrt
{

QUALIFIER_D_H BvhNode::BvhNode()
    : leaf(true), id(kInvalidId), id_left(kInvalidId), id_right(kInvalidId),
      id_object(kInvalidId), area(0), aabb(AABB())
{
}

QUALIFIER_D_H BvhNode::BvhNode(const uint32_t _id)
    : leaf(false), id(_id), id_left(kInvalidId), id_right(kInvalidId),
      id_object(kInvalidId), area(0), aabb(AABB())
{
}

QUALIFIER_D_H BvhNode::BvhNode(const uint32_t _id, const uint32_t _object_id,
                               const AABB &_aabb, const float _area)
    : leaf(true), id(_id), id_left(kInvalidId), id_right(kInvalidId),
      id_object(_object_id), area(_area), aabb(_aabb)
{
}

std::vector<BvhNode> BvhBuilder::Build(const std::vector<AABB> &aabbs,
                                       const std::vector<float> &areas)
{
    std::vector<BvhNode> nodes;
    static BvhBuilder builder;
    try
    {
        nodes = builder.BuildLinearBvh(aabbs, areas);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when build BVH.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
    return nodes;
}

std::vector<BvhNode> BvhBuilder::BuildLinearBvh(const std::vector<AABB> &aabbs,
                                                const std::vector<float> &areas)
{
    uint32_t num_object = static_cast<uint32_t>(aabbs.size());
    map_id_ = std::vector<uint32_t>(num_object);
    for (uint32_t i = 0; i < num_object; ++i)
        map_id_[i] = i;

    aabbs_ = aabbs;
    if (!GenerateMorton())
        throw std::exception("error when generate Morton code.");

    areas_ = areas, nodes_ = {};
    BuildLinearBvhTopDown(0, num_object);
    return nodes_;
}

bool BvhBuilder::GenerateMorton()
{
    uint32_t num_object = static_cast<uint32_t>(aabbs_.size());
    AABB aabb_all;
    for (const AABB &aabb : aabbs_)
        aabb_all += aabb;
    const Vec3 aabb_size = aabb_all.max() - aabb_all.min();

    std::unordered_set<uint64_t> morton_sets;
    mortons_ = std::vector<uint64_t>(num_object);
    bool unique = true;
    Vec3 position_relative;
    for (uint32_t i = 0; i < num_object; ++i)
    {
        position_relative = (aabbs_[i].center() - aabb_all.min()) / aabb_size;
        mortons_[i] = GetMorton3D(position_relative);
        mortons_[i] = (mortons_[i] << 32) | static_cast<uint64_t>(i);
        if (morton_sets.count(mortons_[i]))
        {
            unique = false;
            break;
        }
        morton_sets.insert(mortons_[i]);
    }
    if (unique)
    {
        std::sort(map_id_.begin(), map_id_.end(),
                  [&](const uint64_t id1, const uint64_t id2)
                  { return mortons_[id1] < mortons_[id2]; });
    }

    return unique;
}

uint32_t BvhBuilder::BuildLinearBvhTopDown(const uint32_t begin,
                                           const uint32_t end)
{
    const uint32_t id_node = nodes_.size();
    if (begin + 1 > end)
    {
        return kInvalidId;
    }
    else if (begin + 1 == end)
    {
        nodes_.push_back(BvhNode(id_node, map_id_[begin],
                                 aabbs_[map_id_[begin]],
                                 areas_[map_id_[begin]]));
        return id_node;
    }
    else
    {
        nodes_.push_back(BvhNode(id_node));
        const uint32_t middle = FindSplit(begin, end) + 1;
        nodes_[id_node].id_left = BuildLinearBvhTopDown(begin, middle);
        nodes_[id_node].id_right = BuildLinearBvhTopDown(middle, end);
        nodes_[id_node].area = nodes_[nodes_[id_node].id_left].area +
                               nodes_[nodes_[id_node].id_right].area,
        nodes_[id_node].aabb = nodes_[nodes_[id_node].id_left].aabb +
                               nodes_[nodes_[id_node].id_right].aabb;
        return id_node;
    }
}

uint32_t BvhBuilder::FindSplit(const uint32_t first, const uint32_t last)
{
    // Identical Morton codes => split the range in the middle.
    const uint64_t first_code = mortons_[map_id_[first]],
                   last_code = mortons_[map_id_[last - 1]];
    if (first_code == last_code)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects,
    // using the count-leading-zeros intrinsic.
    const int common_prefix =
        GetConsecutiveHighOrderZeroBitsNum(first_code ^ last_code);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.
    uint32_t split = first; // initial guess
    uint32_t step = last - first;
    do
    {
        step = (step + 1) >> 1;            // exponential decrease
        uint32_t new_split = split + step; // proposed new position

        if (new_split < last)
        {
            const uint64_t split_code = mortons_[map_id_[new_split]];
            const int split_prefix =
                GetConsecutiveHighOrderZeroBitsNum(first_code ^ split_code);
            if (split_prefix > common_prefix)
                split = new_split; // accept proposal
        }
    } while (step > 1);

    return split;
}

} // namespace csrt