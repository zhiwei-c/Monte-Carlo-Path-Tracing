#include "csrt/rtcore/primitives/primitive.cuh"

#include "csrt/utils.cuh"

namespace csrt
{

QUALIFIER_D_H PrimitiveData::PrimitiveData()
    : type(PrimitiveType::kNone), triangle{}, sphere{}
{
}

QUALIFIER_D_H PrimitiveData::PrimitiveData(const PrimitiveData &data)
    : type(data.type)
{
    switch (data.type)
    {
    case PrimitiveType::kNone:
        break;
    case PrimitiveType::kTriangle:
        triangle = data.triangle;
        break;
    case PrimitiveType::kSphere:
        sphere = data.sphere;
        break;
    }
}

QUALIFIER_D_H void PrimitiveData::operator=(const PrimitiveData &data)
{
    type = data.type;
    switch (data.type)
    {
    case PrimitiveType::kNone:
        break;
    case PrimitiveType::kTriangle:
        triangle = data.triangle;
        break;
    case PrimitiveType::kSphere:
        sphere = data.sphere;
        break;
    }
}

QUALIFIER_D_H Primitive::Primitive() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Primitive::Primitive(const uint32_t id, const PrimitiveData &data)
    : id_(id), data_(data)
{
}

QUALIFIER_D_H AABB Primitive::aabb() const
{
    switch (data_.type)
    {
    case PrimitiveType::kTriangle:
        return GetAabbTriangle(data_.triangle);
        break;
    case PrimitiveType::kSphere:
        return GetAabbSphere(data_.sphere);
        break;
    }
    return {};
}

QUALIFIER_D_H bool Primitive::Intersect(Bsdf *bsdf, uint32_t *seed, Ray *ray,
                                        Hit *hit) const
{
    switch (data_.type)
    {
    case PrimitiveType::kTriangle:
        return IntersectTriangle(id_, data_.triangle, bsdf, seed, ray, hit);
        break;
    case PrimitiveType::kSphere:
        return IntersectSphere(id_, data_.sphere, bsdf, seed, ray, hit);
        break;
    }
    return false;
}

QUALIFIER_D_H Hit Primitive::Sample(const float xi_0, const float xi_1) const
{
    switch (data_.type)
    {
    case PrimitiveType::kTriangle:
        return SampleTriangle(id_, data_.triangle, xi_0, xi_1);
        break;
    case PrimitiveType::kSphere:
        return SampleSphere(id_, data_.sphere, xi_0, xi_1);
        break;
    }
    return {};
}

} // namespace csrt