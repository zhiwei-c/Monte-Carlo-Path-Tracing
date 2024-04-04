#ifndef CSRT__RTCORE__PRIMITIVES_PRIMITIVE_HPP
#define CSRT__RTCORE__PRIMITIVES_PRIMITIVE_HPP

#include "../accel/aabb.hpp"
#include "../hit.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

namespace csrt
{

enum class PrimitiveType
{
    kNone,
    kTriangle,
    kSphere,
};

class Bsdf;

struct PrimitiveData
{
    PrimitiveType type;
    union
    {
        TriangleData triangle;
        SphereData sphere;
    };

    QUALIFIER_D_H PrimitiveData();
    QUALIFIER_D_H PrimitiveData(const PrimitiveData &data);
    QUALIFIER_D_H void operator=(const PrimitiveData &data);
};

class Primitive
{
public:
    QUALIFIER_D_H Primitive();
    QUALIFIER_D_H Primitive(const uint32_t id, const PrimitiveData &data);

    QUALIFIER_D_H AABB aabb() const;
    QUALIFIER_D_H bool Intersect(Bsdf *bsdf, uint32_t *seed, Ray *ray,
                                 Hit *hit) const;
    QUALIFIER_D_H Hit Sample(const float xi_0, const float xi_1) const;

private:
    uint32_t id_;
    PrimitiveData data_;
};

} // namespace csrt

#endif