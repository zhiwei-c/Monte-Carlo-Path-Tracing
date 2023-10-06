#include "linear_bvh.cuh"

#include <cstdio>

void LinearBvhBuilder::Build(const std::vector<AABB> &aabb_buffer,
                             std::vector<BvhNode> *bvh_node_buffer)
{
    aabb_buffer_ = aabb_buffer;
    bvh_node_buffer_ = bvh_node_buffer;

    num_objects_ = aabb_buffer.size();
    id_map_ = std::vector<uint64_t>(num_objects_);
    for (uint64_t i = 0; i < num_objects_; ++i)
        id_map_[i] = i;

    if (!GenerateMorton())
    {
        fprintf(stderr, "[error] Build linear BVH failed.\n");
        exit(1);
    }

    *bvh_node_buffer_ = {}, max_depth_ = 0;
    BuildBvhTopDown(0, num_objects_, 0);
    if (max_depth_ >= 64)
    {
        fprintf(stderr, "[error] Build linear BVH failed.\n");
        exit(1);
    }
}

bool LinearBvhBuilder::GenerateMorton()
{
    const AABB aabb_all = GetAabbBottomUpIndexed(0, num_objects_);
    const Vec3 aabb_size = aabb_all.max() - aabb_all.min();
    std::unordered_set<uint64_t> morton_sets;
    morton_buffer_ = std::vector<uint64_t>(num_objects_);
    bool unique = true;
    Vec3 pos_relative;
    for (uint64_t i = 0; i < num_objects_; ++i)
    {
        pos_relative = (aabb_buffer_[i].center() - aabb_all.min()) / aabb_size;
        morton_buffer_[i] = GetMorton3D(pos_relative);
        morton_buffer_[i] = (morton_buffer_[i] << 32) | static_cast<uint32_t>(i);
        if (morton_sets.count(morton_buffer_[i]))
        {
            unique = false;
            break;
        }
        morton_sets.insert(morton_buffer_[i]);
    }
    if (unique)
    {
        std::sort(id_map_.begin(), id_map_.end(), [&](const uint64_t id1, const uint64_t id2)
                  { return morton_buffer_[id1] < morton_buffer_[id2]; });
    }
    return unique;
}

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
uint64_t LinearBvhBuilder::GetMorton3D(const Vec3 &pos)
{
    const float x = fminf(fmaxf(pos.x * 1024.0f, 0.0f), 1023.0f),
                y = fminf(fmaxf(pos.y * 1024.0f, 0.0f), 1023.0f),
                z = fminf(fmaxf(pos.z * 1024.0f, 0.0f), 1023.0f);
    const uint64_t xx = ExpandBits(static_cast<uint64_t>(x)),
                   yy = ExpandBits(static_cast<uint64_t>(y)),
                   zz = ExpandBits(static_cast<uint64_t>(z));
    return xx * 4 + yy * 2 + zz;
}

// Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
uint64_t LinearBvhBuilder::ExpandBits(uint64_t v)
{
    v = (v * ((0x1ull << 16) + 1)) & 0xFF0000FFFF0000FFull;
    v = (v * ((0x1ull << 8) + 1)) & 0x0F00F00F0F00F00Full;
    v = (v * ((0x1ull << 4) + 1)) & 0xC30C30C3C30C30C3ull;
    v = (v * ((0x1ull << 2) + 1)) & 0x4924924949249249ull;
    return v;
}

uint64_t LinearBvhBuilder::BuildBvhTopDown(const uint64_t begin, const uint64_t end,
                                           const uint64_t depth)
{
    const uint64_t id_node = bvh_node_buffer_->size();
    if (begin + 1 > end)
    {
        return kInvalidId;
    }
    else if (begin + 1 == end)
    {
        max_depth_ = depth > max_depth_ ? depth : max_depth_;
        bvh_node_buffer_->push_back(BvhNode(id_node, id_map_[begin], aabb_buffer_[id_map_[begin]]));
        return id_node;
    }
    else
    {
        const AABB aabb_current = GetAabbBottomUpIndexed(begin, end);
        bvh_node_buffer_->push_back(BvhNode(id_node, aabb_current));

        const uint64_t middle = FindSplit(begin, end) + 1;
        (*bvh_node_buffer_)[id_node].id_left = BuildBvhTopDown(begin, middle, depth + 1);
        (*bvh_node_buffer_)[id_node].id_right = BuildBvhTopDown(middle, end, depth + 1);
        return id_node;
    }
}

uint64_t LinearBvhBuilder::FindSplit(const uint64_t first, const uint64_t last)
{
    // Identical Morton codes => split the range in the middle.

    const uint64_t first_code = morton_buffer_[id_map_[first]],
                   last_code = morton_buffer_[id_map_[last - 1]];

    if (first_code == last_code)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects, using the count-leading-zeros intrinsic.

    const uint64_t common_prefix = GetConsecutiveHighOrderZeroBitsNum(first_code ^ last_code);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    uint64_t split = first; // initial guess
    uint64_t step = last - first;
    do
    {
        step = (step + 1) >> 1;            // exponential decrease
        uint64_t new_split = split + step; // proposed new position

        if (new_split < last)
        {
            const uint64_t split_code = morton_buffer_[id_map_[new_split]];
            const uint64_t split_prefix = GetConsecutiveHighOrderZeroBitsNum(first_code ^ split_code);
            if (split_prefix > common_prefix)
                split = new_split; // accept proposal
        }
    } while (step > 1);

    return split;
}

uint64_t LinearBvhBuilder::GetConsecutiveHighOrderZeroBitsNum(const uint64_t n)
{
    uint64_t count = 0;
    for (int i = 0; i < 64; ++i)
    {
        if ((n >> (63 - i)) & 0x1)
            break;
        else
            ++count;
    }
    return count;
}