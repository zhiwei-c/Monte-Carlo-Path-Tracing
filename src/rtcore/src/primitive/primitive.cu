#include "primitive.cuh"

#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H Primitive::Data::Data()
    : type(Primitive::Type::kNone), triangle{}, sphere{}
{
}

QUALIFIER_D_H Primitive::Data::Data(const Primitive::Data &Data)
    : type(Data.type)
{
    switch (Data.type)
    {
    case Primitive::Type::kNone:
        break;
    case Primitive::Type::kTriangle:
        triangle = Data.triangle;
        break;
    case Primitive::Type::kSphere:
        sphere = Data.sphere;
        break;
    }
}

QUALIFIER_D_H void Primitive::Data::operator=(const Primitive::Data &Data)
{
    type = Data.type;
    switch (Data.type)
    {
    case Primitive::Type::kNone:
        break;
    case Primitive::Type::kTriangle:
        triangle = Data.triangle;
        break;
    case Primitive::Type::kSphere:
        sphere = Data.sphere;
        break;
    }
}

QUALIFIER_D_H Primitive::Primitive() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Primitive::Primitive(const uint32_t id,
                                   const Primitive::Data &Data)
    : id_(id), data_(Data)
{
}

QUALIFIER_D_H AABB Primitive::aabb() const
{
    switch (data_.type)
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
    switch (data_.type)
    {
    case Primitive::Type::kTriangle:
        IntersectTriangle(ray, hit);
        break;
    case Primitive::Type::kSphere:
        IntersectSphere(ray, hit);
        break;
    }
}

QUALIFIER_D_H Hit Primitive::Sample(const float xi_0, const float xi_1) const
{
    switch (data_.type)
    {
    case Primitive::Type::kTriangle:
        return SampleTriangle(xi_0, xi_1);
    case Primitive::Type::kSphere:
        return SampleSphere(xi_0, xi_1);
    default:
        return {};
    }
}

} // namespace rt