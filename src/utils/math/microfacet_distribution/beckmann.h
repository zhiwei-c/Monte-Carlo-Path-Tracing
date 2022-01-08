#pragma once

#include <memory>

#include "microfacet_distribution.h"
#include "../math.h"

NAMESPACE_BEGIN(simple_renderer)

class Beckmann : public MicrofacetDistribution
{
public:
    Beckmann(Float alpha_u, Float alpha_v)
        : MicrofacetDistribution(MicrofacetDistribType::kBeckmann, alpha_u, alpha_v) {}

    Vector3 Sample(const Vector3 &normal_macro, const Vector2 &sample) const
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
            Float ratio = alpha_v_ / alpha_u_,
                  tmp = ratio * std::tan((2 * kPi) * sample.y);
            cos_phi = 1 / std::sqrt(tmp * tmp + 1);
            if (std::fabs(sample.y - .5) - .25 > 0)
                cos_phi = -cos_phi;
            sin_phi = cos_phi * tmp;
            alpha_2 = 1 / (std::pow(cos_phi / alpha_u_, 2) +
                           std::pow(sin_phi / alpha_v_, 2));
        }

        auto cos_theta = 1 / std::sqrt(1 - alpha_2 * std::log(1 - sample.x)),
             sin_theta = std::sqrt(1 - cos_theta * cos_theta);

        auto normal_micro_local = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        return ToWorld(normal_micro_local, normal_macro);
    }

    Float Eval(const Vector3 &normal_micro, const Vector3 &normal_macro) const
    {
        auto cos_theta = glm::dot(normal_macro, normal_micro);

        if (cos_theta <= 0)
            return 0;

        auto cos_theta_2 = std::pow(cos_theta, 2),
             tan_theta_2 = (1 - cos_theta_2) / cos_theta_2,
             cos_theta_4 = std::pow(cos_theta, 4);
        auto alpha_2 = alpha_u_ * alpha_v_;

        Float result = 0;
        if (isotropic_)
            result = std::exp(-tan_theta_2 / alpha_2) / (kPi * alpha_2 * cos_theta_4);
        else
        {
            auto dir = ToLocal(normal_micro, normal_macro);
            result = std::exp(-(std::pow(dir.x / alpha_u_, 2) + std::pow(dir.y / alpha_v_, 2)) / cos_theta_2) / (kPi * alpha_2 * cos_theta_4);
        }
        return (result * cos_theta > 1e-20) ? result : 0;
    }

    Float SmithG1(const Vector3 &v, const Vector3 &normal_micro, const Vector3 &normal_macro) const
    {
        auto cos_theta_v_n = glm::dot(v, normal_macro);
        auto cos_theta_v_m = glm::dot(v, normal_micro);
        if (cos_theta_v_n * cos_theta_v_m <= 0)
            return 0;

        if (std::fabs(cos_theta_v_n - 1) < kEpsilon)
            return 1;

        Float a = 0;
        if (isotropic_)
        {
            auto tan_theta_v_n = std::sqrt(1 - std::pow(cos_theta_v_n, 2)) / cos_theta_v_n;
            a = 1 / (alpha_u_ * tan_theta_v_n);
        }
        else
        {
            auto dir = ToLocal(v, normal_macro);
            Float xy_alpha_2 = std::pow(alpha_u_ * dir.x, 2) + std::pow(alpha_v_ * dir.y, 2),
                  tan_theta_alpha_2 = xy_alpha_2 / std::pow(dir.z, 2);
            a = 1 / std::sqrt(tan_theta_alpha_2);
        }

        if (a < 1.6)
        {
            auto a_2 = std::pow(a, 2);
            return (3.535 * a + 2.181 * a_2) / (1 + 2.276 * a + 2.577 * a_2);
        }
        else
        {
            return 1;
        }
    }
};

NAMESPACE_END(simple_renderer)