#pragma once

#include <memory>

#include "../core/microfacet_distribution_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief Beckmann 微表面分布派生类
class Beckmann : public MicrofacetDistribution
{
public:
    ///\brief Beckmann 微表面分布
    ///\param alpha_u 沿切线（tangent）方向的粗糙度
    ///\param alpha_v 沿沿副切线（bitangent）方向的粗糙度
    Beckmann(Float alpha_u, Float alpha_v) : MicrofacetDistribution(alpha_u, alpha_v) {}

    ///\brief 抽样微表面法线
    std::pair<Vector3, Float> Sample(const Vector3 &normal_macro, const Vector2 &sample) const override
    {
        Float sin_phi = 0, cos_phi = 0, alpha_2 = 0;
        if (isotropic_)
        {
            Float phi = 2.0 * kPi * sample.y;
            cos_phi = std::cos(phi);
            sin_phi = std::sin(phi);
            alpha_2 = alpha_u_ * alpha_u_;
        }
        else
        {
            Float phi = std::atan(alpha_v_ / alpha_u_ * std::tan(kPi + 2 * kPi * sample.y)) + kPi * std::floor(2 * sample.y + 0.5);
            cos_phi = std::cos(phi);
            sin_phi = std::sin(phi);
            alpha_2 = 1.0 / (Sqr(cos_phi / alpha_u_) + Sqr(sin_phi / alpha_v_));
        }
        Float cos_theta = 1.0 / std::sqrt(1.0 - alpha_2 * std::log(1.0 - sample.x)),
              sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        auto normal_micro_local = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        Float pdf = (1.0 - sample.x) / (kPi * alpha_u_ * alpha_v_ * std::pow(cos_theta, 3));
        return {ToWorld(normal_micro_local, normal_macro), pdf};
    }

    ///\brief 计算给定微表面法线余弦加权的概率
    Float Pdf(const Vector3 &normal_micro, const Vector3 &normal_macro) const override
    {
        Float cos_theta = glm::dot(normal_macro, normal_micro);
        if (cos_theta <= 0)
            return 0;
        Float cos_theta_2 = std::pow(cos_theta, 2),
              tan_theta_2 = (1.0 - cos_theta_2) / cos_theta_2,
              cos_theta_3 = std::pow(cos_theta, 3);
        Float alpha_2 = alpha_u_ * alpha_v_;
        if (isotropic_)
            return std::exp(-tan_theta_2 / alpha_2) / (kPi * alpha_2 * cos_theta_3);
        else
        {
            Vector3 dir = ToLocal(normal_micro, normal_macro);
            return std::exp(-(Sqr(dir.x / alpha_u_) + Sqr(dir.y / alpha_v_)) / cos_theta_2) / (kPi * alpha_2 * cos_theta_3);
        }
    }

    ///\brief 计算给定参数的阴影-遮蔽系数
    Float SmithG1(const Vector3 &v, const Vector3 &normal_micro, const Vector3 &normal_macro) const override
    {
        Float cos_v_n = glm::dot(v, normal_macro);
        if (cos_v_n * glm::dot(v, normal_micro) <= 0)
            return 0;

        if (std::abs(cos_v_n - 1) < kEpsilon)
            return 1;
        Float a = 0;
        if (isotropic_)
        {
            Float tan_theta_v_n = std::sqrt(1.0 - std::pow(cos_v_n, 2)) / cos_v_n;
            a = 1.0 / (alpha_u_ * tan_theta_v_n);
        }
        else
        {
            Vector3 dir = ToLocal(v, normal_macro);
            Float xy_alpha_2 = std::pow(alpha_u_ * dir.x, 2) + std::pow(alpha_v_ * dir.y, 2),
                  tan_theta_alpha_2 = xy_alpha_2 / std::pow(dir.z, 2);
            a = 1.0 / std::sqrt(tan_theta_alpha_2);
        }

        if (a < 1.6)
        {
            Float a_2 = std::pow(a, 2);
            return (3.535 * a + 2.181 * a_2) / (1 + 2.276 * a + 2.577 * a_2);
        }
        else
            return 1;
    }
};

NAMESPACE_END(raytracer)