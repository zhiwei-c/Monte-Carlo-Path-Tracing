#pragma once

#include "../tensor/tensor.cuh"
#include "../renderer/ray.cuh"

class AABB
{
public:
    QUALIFIER_DEVICE AABB();
    QUALIFIER_DEVICE AABB(const Vec3 &min, const Vec3 &max);

    QUALIFIER_DEVICE bool Intersect(const Ray &ray) const;

    QUALIFIER_DEVICE AABB &operator+=(const Vec3 &v);
    QUALIFIER_DEVICE AABB &operator+=(const AABB &aabb);

    QUALIFIER_DEVICE Vec3 min() const { return min_; }
    QUALIFIER_DEVICE Vec3 max() const { return max_; }
    QUALIFIER_DEVICE Vec3 center() const { return center_; }

private:
    Vec3 min_;
    Vec3 max_;
    Vec3 center_;
};

QUALIFIER_DEVICE AABB operator+(const AABB &a, const AABB &b);