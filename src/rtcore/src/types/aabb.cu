#include "types.cuh"

#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H AABB::AABB() : min_{kMaxFloat}, max_{kLowestFloat} {}

QUALIFIER_D_H AABB::AABB(const Vec3 &min, const Vec3 &max) : min_(min), max_(max) {}

QUALIFIER_D_H AABB &AABB::operator+=(const Vec3 &v)
{
    min_ = Min(v, min_);
    max_ = Max(v, max_);
    return *this;
}

QUALIFIER_D_H AABB &AABB::operator+=(const AABB &aabb)
{
    min_ = Min(aabb.min(), min_);
    max_ = Max(aabb.max(), max_);
    return *this;
}

QUALIFIER_D_H bool AABB::Intersect(Ray *ray) const
{
    const Vec3 t_min = (min_ - ray->origin) * ray->dir_rcp,
               t_max = (max_ - ray->origin) * ray->dir_rcp;
    float t_enter = ray->t_min, t_exit = ray->t_max;
    for (int i = 0; i < 3; ++i)
    {
        if (ray->dir_rcp[i] > 0)
        {
            t_enter = fmaxf(t_enter, t_min[i]);
            t_exit = fminf(t_exit, t_max[i]);
        }
        else
        {
            t_enter = fmaxf(t_enter, t_max[i]);
            t_exit = fminf(t_exit, t_min[i]);
        }
    }
    return t_enter <= t_exit;
}

QUALIFIER_D_H AABB operator+(const AABB &a, const AABB &b)
{
    return AABB(Min(a.min(), b.min()), Max(a.max(), b.max()));
}

} // namespace rt