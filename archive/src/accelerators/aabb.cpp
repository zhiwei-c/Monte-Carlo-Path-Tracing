#include "aabb.hpp"

#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

AABB::AABB()
    : min_(dvec3(kMaxDouble)),
      max_(dvec3(kLowestDouble)),
      center_(dvec3(0))
{
}

AABB::AABB(const dvec3 &min, const dvec3 &max)
    : min_(min),
      max_(max),
      center_((min_ + max_) * 0.5)
{
}

bool AABB::Intersect(const Ray &ray) const
{
    double t_enter = kLowestDouble,
           t_exit = kMaxDouble;
    for (int dim = 0; dim < 3; ++dim)
    {
        double t_min = (min_[dim] - ray.origin()[dim]) * ray.dir_rcp()[dim],
               t_max = (max_[dim] - ray.origin()[dim]) * ray.dir_rcp()[dim];
        if (ray.dir()[dim] < 0)
        {
            std::swap(t_min, t_max);
        }
        t_enter = std::max(t_min, t_enter);
        t_exit = std::min(t_max, t_exit);
    }
    t_exit *= kAabbErrorBound;
    return 0.0 < t_exit && t_enter < t_exit && t_enter < ray.t_max();
}

dvec3 AABB::center() const
{
    return center_;
}

AABB AABB::operator+(const AABB &b) const
{
    return AABB(glm::min(min_, b.min()), glm::max(max_, b.max()));
}

AABB &AABB::operator+=(const AABB &rhs)
{
    min_ = glm::min(min_, rhs.min());
    max_ = glm::max(max_, rhs.max());
    center_ = (min_ + max_) * 0.5;
    return *this;
}

AABB &AABB::operator+=(const dvec3 &rhs)
{
    min_ = glm::min(min_, rhs);
    max_ = glm::max(max_, rhs);
    center_ = (min_ + max_) * 0.5;
    return *this;
}

NAMESPACE_END(raytracer)