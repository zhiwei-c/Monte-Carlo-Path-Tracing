#pragma once

#include <memory>

#include "../core/microfacet_distribution_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief GGX 微表面分布派生类
class GGX : public MicrofacetDistribution
{
public:
    ///\brief GGX 微表面分布
    ///\param alpha_u 沿切线（tangent）方向的粗糙度
    ///\param alpha_v 沿沿副切线（bitangent）方向的粗糙度
    GGX(Float alpha_u, Float alpha_v)
        : MicrofacetDistribution(alpha_u, alpha_v) {}

    ///\brief 抽样微表面法线
    std::pair<Vector3, Float> Sample(const Vector3 &normal_macro, const Vector2 &sample) const
    {
        Float sin_phi, cos_phi, alpha_2;
        if (isotropic_)
        {
            auto phi = 2 * kPi * sample.y;
            cos_phi = std::cos(phi);
            sin_phi = std::sin(phi);
            alpha_2 = alpha_u_ * alpha_u_;
        }
        else
        {
            auto phi = std::atan(alpha_v_ / alpha_u_ * std::tan(kPi + 2 * kPi * sample.y)) + kPi * std::floor(2 * sample.y + 0.5);
            cos_phi = std::cos(phi);
            sin_phi = std::sin(phi);
            alpha_2 = 1 / (Sqr(cos_phi / alpha_u_) + Sqr(sin_phi / alpha_v_));
        }
        auto tan_theta_2 = alpha_2 * sample.x / (1 - sample.x);
        auto cos_theta = 1 / std::sqrt(1 + tan_theta_2),
             sin_theta = std::sqrt(1 - cos_theta * cos_theta);

        auto normal_micro_local = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        auto pdf = 1 / (kPi * alpha_u_ * alpha_v_ * std::pow(cos_theta, 3) * Sqr(1 + tan_theta_2 / alpha_2));
        return {ToWorld(normal_micro_local, normal_macro), pdf};
    }

    ///\brief 计算给定微表面法线的概率
    Float Pdf(const Vector3 &normal_micro, const Vector3 &normal_macro) const
    {
        auto cos_theta = glm::dot(normal_macro, normal_micro);

        if (cos_theta <= 0)
            return 0;

        auto sin_theta = std::sqrt(1 - std::pow(cos_theta, 2)),
             tan_theta_2 = std::pow(sin_theta / cos_theta, 2),
             cos_theta_3 = std::pow(cos_theta, 3);
        auto alpha_2 = alpha_u_ * alpha_v_;

        Float result = 0;
        if (isotropic_)
            result = alpha_2 / (kPi * cos_theta_3 * std::pow(alpha_2 + tan_theta_2, 2));
        else
        {
            auto dir = ToLocal(normal_micro, normal_macro);
            result = cos_theta / (kPi * alpha_2 *
                                  std::pow(
                                      std::pow(dir.x / alpha_u_, 2) + std::pow(dir.y / alpha_v_, 2) + std::pow(dir.z, 2),
                                      2));
        }
        return result;
    }

    ///\brief 计算给定参数的阴影-遮蔽系数
    Float SmithG1(const Vector3 &v, const Vector3 &normal_micro, const Vector3 &normal_macro) const
    {
        auto cos_v_n = glm::dot(v, normal_macro);
        auto cos_v_m = glm::dot(v, normal_micro);
        if (cos_v_n * cos_v_m <= 0)
            return 0;

        if (std::fabs(cos_v_n - 1) < kEpsilon)
            return 1;

        Float result = 0;
        if (isotropic_)
        {
            auto cos_v_n_2 = std::pow(cos_v_n, 2);
            auto tan_v_n_2 = (1 - cos_v_n_2) / cos_v_n_2;
            auto alpha_2 = alpha_u_ * alpha_u_;
            result = 2 / (1 + std::sqrt(1 + alpha_2 * tan_v_n_2));
        }
        else
        {
            auto dir = ToLocal(v, normal_macro);
            Float xy_alpha_2 = std::pow(alpha_u_ * dir.x, 2) + std::pow(alpha_v_ * dir.y, 2),
                  tan_v_n_alpha_2 = xy_alpha_2 / std::pow(dir.z, 2);
            result = 2 / (1 + std::sqrt(1 + tan_v_n_alpha_2));
        }

        return result;
    }
};

NAMESPACE_END(simple_renderer)