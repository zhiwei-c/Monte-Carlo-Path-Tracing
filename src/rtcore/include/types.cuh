#pragma once

#include "tensor.cuh"

#include <vector>

namespace rt
{

struct Ray
{
    float t_min;
    float t_max;
#ifdef WATERTIGHT_TRIANGLES
    int k[3];
    Vec3 shear;
#endif
    Vec3 origin;
    Vec3 dir;
    Vec3 dir_rcp;

    QUALIFIER_D_H Ray();
    QUALIFIER_D_H Ray(const Vec3 &_origin, const Vec3 &_dir);
};

struct Hit
{
    bool valid;
    bool inside;
    uint32_t id_instance;
    uint32_t id_primitve;
    Vec2 texcoord;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;

    QUALIFIER_D_H Hit();
    QUALIFIER_D_H Hit(const bool _id_primitve, const Vec2 &_texcoord, const Vec3 &_position,
                      const Vec3 &_normal);
    QUALIFIER_D_H Hit(const bool _id_primitve, const bool _inside, const Vec2 &_texcoord,
                      const Vec3 &_position, const Vec3 &_normal, const Vec3 &_tangent,
                      const Vec3 &_bitangent);
};

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

} // namespace rt