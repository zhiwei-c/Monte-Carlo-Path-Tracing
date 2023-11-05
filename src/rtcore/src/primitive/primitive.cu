#include "primitive.cuh"

#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H Primitive::Info::Info() : type(Primitive::Type::kNone), triangle{}, sphere{} {}

QUALIFIER_D_H Primitive::Info::Info(const Primitive::Info &info) : type(info.type)
{
    switch (info.type)
    {
    case Primitive::Type::kNone:
        break;
    case Primitive::Type::kTriangle:
        triangle = info.triangle;
        break;
    case Primitive::Type::kSphere:
        sphere = info.sphere;
        break;
    }
}

QUALIFIER_D_H void Primitive::Info::operator=(const Primitive::Info &info)
{
    type = info.type;
    switch (info.type)
    {
    case Primitive::Type::kNone:
        break;
    case Primitive::Type::kTriangle:
        triangle = info.triangle;
        break;
    case Primitive::Type::kSphere:
        sphere = info.sphere;
        break;
    }
}

QUALIFIER_D_H Primitive::Primitive() : id_primitive_(kInvalidId), geom_{} {}

QUALIFIER_D_H Primitive::Primitive(const uint32_t id_primitive, const Primitive::Info &info)
    : id_primitive_(id_primitive), geom_(info)
{
}

QUALIFIER_D_H AABB Primitive::aabb() const
{
    switch (geom_.type)
    {
    case Primitive::Type::kTriangle:
        return GetAabbTriangle();
    case Primitive::Type::kSphere:
        return GetAabbSphere();
    default:
        return {};
    }
}

QUALIFIER_D_H void Primitive::Intersect(Ray *ray, Hit *hit) const
{
    switch (geom_.type)
    {
    case Primitive::Type::kTriangle:
        IntersectTriangle(ray, hit);
        break;
    case Primitive::Type::kSphere:
        IntersectSphere(ray, hit);
        break;
    }
}

QUALIFIER_D_H Hit Primitive::Sample(const float xi_0, const float xi_1, const float xi_2) const
{
    switch (geom_.type)
    {
    case Primitive::Type::kTriangle:
        return SampleTriangle(xi_0, xi_1, xi_2);
    case Primitive::Type::kSphere:
        return SampleSphere(xi_0, xi_1, xi_2);
    default:
        return {};
    }
}

} // namespace rt