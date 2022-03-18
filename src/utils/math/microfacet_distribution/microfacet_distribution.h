#pragma once

#include <utility>

#include "../../global.h"

NAMESPACE_BEGIN(simple_renderer)

enum class MicrofacetDistribType
{
    kBeckmann,
    kGgx
};

class MicrofacetDistribution
{
public:
    virtual ~MicrofacetDistribution() {}

    virtual std::pair<Vector3, Float> Sample(const Vector3 &normal_macro, const Vector2 &sample) const = 0;

    virtual Float Pdf(const Vector3 &normal_micro, const Vector3 &normal_macro) const = 0;

    virtual Float SmithG1(const Vector3 &v, const Vector3 &normal_micro, const Vector3 &normal_macro) const = 0;

    void ScaleAlpha(Float value)
    {
        alpha_u_ *= value;
        alpha_v_ *= value;
    }

    MicrofacetDistribType type() const { return type_; }

protected:
    bool isotropic_;
    MicrofacetDistribType type_;
    Float alpha_u_, alpha_v_;

    MicrofacetDistribution(MicrofacetDistribType type, Float alpha_u, Float alpha_v)
        : type_(type), isotropic_(alpha_u == alpha_v), alpha_u_(alpha_u), alpha_v_(alpha_v) {}
};

NAMESPACE_END(simple_renderer)