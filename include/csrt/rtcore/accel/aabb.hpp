#ifndef CSRT__RTCORE__ACCEL_AABB_HPP
#define CSRT__RTCORE__ACCEL_AABB_HPP

#include "../../tensor.hpp"
#include "../ray.hpp"

namespace csrt
{

class AABB
{
public:
    QUALIFIER_D_H AABB();
    QUALIFIER_D_H AABB(const Vec3 &min, const Vec3 &max);

    QUALIFIER_D_H AABB &operator+=(const Vec3 &v);
    QUALIFIER_D_H AABB &operator+=(const AABB &aabb);

    QUALIFIER_D_H Vec3 min() const { return min_; }
    QUALIFIER_D_H Vec3 max() const { return max_; }
    QUALIFIER_D_H Vec3 center() const { return (min_ + max_) * 0.5f; }
    QUALIFIER_D_H bool Intersect(Ray *ray) const;

private:
    Vec3 min_;
    Vec3 max_;
};

QUALIFIER_D_H AABB operator+(const AABB &a, const AABB &b);

} // namespace csrt

#endif