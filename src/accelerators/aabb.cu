#include "aabb.cuh"

#include "../utils/math.cuh"
#include "../utils/math.cuh"

QUALIFIER_DEVICE AABB::AABB()
    : min_(Vec3(kMaxFloat)), max_(Vec3(kLowestFloat)), center_(Vec3(0))
{
}

QUALIFIER_DEVICE AABB::AABB(const Vec3 &min, const Vec3 &max)
    : min_(min), max_(max), center_((min + max) * 0.5f)
{
}

QUALIFIER_DEVICE bool AABB::Intersect(const Ray &ray) const
{
    float t_enter = kLowestFloat, t_exit = kMaxFloat,
          t_min, t_max, t_temp;
    for (int dim = 0; dim < 3; ++dim)
    {
        t_min = (min_[dim] - ray.origin[dim]) * ray.dir_inv[dim],
        t_max = (max_[dim] - ray.origin[dim]) * ray.dir_inv[dim];
        if (ray.dir_inv[dim] < 0.0f)
        {
            t_temp = t_min;
            t_min = t_max;
            t_max = t_temp;
        }
        t_enter = fmaxf(t_min, t_enter);
        t_exit = fminf(t_max, t_exit);
    }
    t_exit *= kAabbErrorBound;
    return 0.0f < t_exit && t_enter < t_exit;
}

QUALIFIER_DEVICE AABB &AABB::operator+=(const Vec3 &v)
{
    min_ = Min(min_, v);
    max_ = Max(max_, v);
    center_ = (min_ + max_) * 0.5f;
    return *this;
}

QUALIFIER_DEVICE AABB &AABB::operator+=(const AABB &aabb)
{
    min_ = Min(min_, aabb.min());
    max_ = Max(max_, aabb.max());
    center_ = (min_ + max_) * 0.5f;
    return *this;
}

QUALIFIER_DEVICE AABB operator+(const AABB &a, const AABB &b)
{
    return AABB(Min(a.min(), b.min()), Max(a.max(), b.max()));
}