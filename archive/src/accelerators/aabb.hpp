#pragma once

#include <array>

#include "../core/ray.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

// 轴对齐包围盒（axis align bounding box）
class AABB
{
public:
    AABB();
    AABB(const dvec3 &min, const dvec3 &max);

    bool Intersect(const Ray &ray) const;

    const dvec3 &min() const { return min_; }
    const dvec3 &max() const { return max_; }

    dvec3 center() const;
    AABB operator+(const AABB &b) const;
    AABB &operator+=(const AABB &rhs);
    AABB &operator+=(const dvec3 &rhs);

private:
    dvec3 min_;    //轴对齐包围盒的底边界
    dvec3 max_;    //轴对齐包围盒的顶边界
    dvec3 center_; //轴对齐包围盒的中心
};

NAMESPACE_END(raytracer)