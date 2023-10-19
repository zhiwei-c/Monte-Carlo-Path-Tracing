#include "linear_bvh.cuh"

#include <cstdio>

void LinearBvhBuilder::Build(uint32_t num_object, AABB *aabb_buffer,
                             std::vector<BvhNode> *bvh_node_buffer)
{
    num_object_ = num_object;
    aabb_buffer_ = aabb_buffer;
    bvh_node_buffer_ = bvh_node_buffer;

    id_map_ = std::vector<uint32_t>(num_object_);
    for (uint32_t i = 0; i < num_object_; ++i)
        id_map_[i] = i;

    if (!GenerateMorton())
    {
        fprintf(stderr, "[error] Build linear BVH failed.\n");
        exit(1);
    }

    *bvh_node_buffer_ = {}, max_depth_ = 0;
    BuildBvhTopDown(0, num_object_, 0);
    if (max_depth_ > 127)
    {
        fprintf(stderr, "[error] Build linear BVH failed.\n");
        exit(1);
    }
}

bool LinearBvhBuilder::GenerateMorton()
{
    const AABB aabb_all = GetAabbBottomUpIndexed(0, num_object_);
    const Vec3 aabb_size = aabb_all.max() - aabb_all.min();
    std::unordered_set<uint64_t> morton_sets;
    morton_buffer_ = std::vector<uint64_t>(num_object_);
    bool unique = true;
    Vec3 position_relative;
    for (uint32_t i = 0; i < num_object_; ++i)
    {
        position_relative = (aabb_buffer_[i].center() - aabb_all.min()) / aabb_size;
        morton_buffer_[i] = GetMorton3D(position_relative);
        morton_buffer_[i] = (morton_buffer_[i] << 32) | static_cast<uint64_t>(i);
        if (morton_sets.count(morton_buffer_[i]))
        {
            unique = false;
            break;
        }
        morton_sets.insert(morton_buffer_[i]);
    }
    if (unique)
    {
        std::sort(id_map_.begin(), id_map_.end(), [&](const uint32_t id1, const uint32_t id2)
                  { return morton_buffer_[id1] < morton_buffer_[id2]; });
    }
    return unique;
}

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
uint32_t LinearBvhBuilder::GetMorton3D(const Vec3 &pos)
{
    const float x = fminf(fmaxf(pos.x * 1024.0f, 0.0f), 1023.0f),
                y = fminf(fmaxf(pos.y * 1024.0f, 0.0f), 1023.0f),
                z = fminf(fmaxf(pos.z * 1024.0f, 0.0f), 1023.0f);
    const uint32_t xx = ExpandBits(static_cast<uint32_t>(x)),
                   yy = ExpandBits(static_cast<uint32_t>(y)),
                   zz = ExpandBits(static_cast<uint32_t>(z));
    return xx * 4 + yy * 2 + zz;
}

// Expands a 10-bit integer into 30 bits by inserting 2 zeros before each bit.
uint32_t LinearBvhBuilder::ExpandBits(uint32_t v)
{
    v = (v * ((0x1ul << 16) + 1)) & 0xFF0000FFul;
    v = (v * ((0x1ul << 8) + 1)) & 0x0F00F00Ful;
    v = (v * ((0x1ul << 4) + 1)) & 0xC30C30C3ul;
    v = (v * ((0x1ul << 2) + 1)) & 0x49249249ul;
    return v;
}

uint32_t LinearBvhBuilder::BuildBvhTopDown(const uint32_t begin, const uint32_t end,
                                           const uint32_t depth)
{
    const uint32_t id_node = bvh_node_buffer_->size();
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

        const uint32_t middle = FindSplit(begin, end) + 1;
        (*bvh_node_buffer_)[id_node].id_left = BuildBvhTopDown(begin, middle, depth + 1);
        (*bvh_node_buffer_)[id_node].id_right = BuildBvhTopDown(middle, end, depth + 1);
        return id_node;
    }
}

uint32_t LinearBvhBuilder::FindSplit(const uint32_t first, const uint32_t last)
{
    // Identical Morton codes => split the range in the middle.

    const uint64_t first_code = morton_buffer_[id_map_[first]],
                   last_code = morton_buffer_[id_map_[last - 1]];

    if (first_code == last_code)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects, using the count-leading-zeros intrinsic.

    const int common_prefix = GetConsecutiveHighOrderZeroBitsNum(first_code ^ last_code);

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
            const uint64_t split_code = morton_buffer_[id_map_[new_split]];
            const int split_prefix = GetConsecutiveHighOrderZeroBitsNum(
                first_code ^ split_code);
            if (split_prefix > common_prefix)
                split = new_split; // accept proposal
        }
    } while (step > 1);

    return split;
}

int LinearBvhBuilder::GetConsecutiveHighOrderZeroBitsNum(const uint64_t n)
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