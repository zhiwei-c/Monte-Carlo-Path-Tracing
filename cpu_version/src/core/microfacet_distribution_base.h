#pragma once

#include <utility>

#include "../utils/math/math_base.h"
#include "../utils/global.h"

NAMESPACE_BEGIN(raytracer)

///\brief 微表面分布类型
enum class MicrofacetDistribType
{
    kBeckmann,
    kGgx
};

///\brief 微表面分布基类
class MicrofacetDistribution
{
public:
    virtual ~MicrofacetDistribution() {}

    ///\brief 抽样微表面法线
    virtual std::pair<Vector3, Float> Sample(const Vector3 &normal_macro, const Vector2 &sample) const = 0;

    ///\brief 计算给定微表面法线的概率
    virtual Float Pdf(const Vector3 &normal_micro, const Vector3 &normal_macro) const = 0;

    ///\brief 计算给定参数的阴影-遮蔽系数
    virtual Float SmithG1(const Vector3 &v, const Vector3 &normal_micro, const Vector3 &normal_macro) const = 0;

    ///\brief 放缩材质的粗糙程度
    void ScaleAlpha(Float value)
    {
        alpha_u_ *= value;
        alpha_v_ *= value;
    }

protected:
    bool isotropic_;             //是否各项同性
    MicrofacetDistribType type_; //微表面分布类型
    Float alpha_u_;              //沿切线（tangent）方向的粗糙度
    Float alpha_v_;              //沿副切线（bitangent）方向的粗糙度

    MicrofacetDistribution(Float alpha_u, Float alpha_v)
        : isotropic_(alpha_u == alpha_v), alpha_u_(alpha_u), alpha_v_(alpha_v)
    {
    }
};

NAMESPACE_END(raytracer)