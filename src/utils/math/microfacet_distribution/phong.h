#pragma once

#include <memory>

#include "microfacet_distribution.h"
#include "../math.h"

NAMESPACE_BEGIN(simple_renderer)

class Phong : public MicrofacetDistribution
{
public:
    Phong(Float alpha_u, Float alpha_v)
        : MicrofacetDistribution(MicrofacetDistribType::kPhong, alpha_u, alpha_v) {}

    Vector3 Sample(const Vector3 &normal_macro) const
    {
        Float sin_phi, cos_phi, alpha_2;

        auto u_1 = UniformFloat();
        auto u_2 = UniformFloat();

        if (isotropic_)
        {
            auto phi = 2 * kPi * u_2;
            cos_phi = std::cos(phi);
            sin_phi = std::sin(phi);
            alpha_2 = alpha_u_ * alpha_u_;
        }
        else
        {
            Float ratio = alpha_v_ / alpha_u_,
                  tmp = ratio * std::tan((2 * kPi) * u_2);
            cos_phi = 1 / std::sqrt(tmp * tmp + 1);
            if (std::fabs(u_2 - .5) - .25 > 0)
                cos_phi = -cos_phi;
            sin_phi = cos_phi * tmp;
            alpha_2 = 1 / (std::pow(cos_phi / alpha_u_, 2) +
                           std::pow(sin_phi / alpha_v_, 2));
        }
        auto alpha = 2 / alpha_2 - 2;

        auto cos_theta = std::pow(u_1, 1 / (alpha + 2)),
             sin_theta = std::sqrt(1 - cos_theta * cos_theta);

        auto normal_micro_local = Vector3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        return ToWorld(normal_micro_local, normal_macro);
    }

    Float Eval(const Vector3 &normal_micro, const Vector3 &normal_macro) const
    {
        auto cos_theta = glm::dot(normal_macro, normal_micro);

        if (cos_theta <= 0)
            return 0;

        Float alpha_2;
        if (isotropic_)
            alpha_2 = alpha_u_ * alpha_v_;
        else
        {
            auto dir = ToLocal(normal_micro, normal_macro);
            auto sin_theta = std::sqrt(1 - dir.z * dir.z);
            auto cos_phi = dir.x / sin_theta,
                 sin_phi = dir.y / sin_theta;
            alpha_2 = 1 / (std::pow(cos_phi / alpha_u_, 2) +
                           std::pow(sin_phi / alpha_v_, 2));
        }
        auto alpha_p = 2 / alpha_2 - 2;
        auto result = (alpha_p + 2) * 0.5 * kPiInv * std::pow(cos_theta, alpha_p);

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

        Float alpha_2;
        if (isotropic_)
        {
            alpha_2 = alpha_u_ * alpha_v_;
        }
        else
        {
            auto dir = ToLocal(normal_micro, normal_macro);
            auto sin_theta = std::sqrt(1 - dir.z * dir.z);
            auto cos_phi = dir.x / sin_theta,
                 sin_phi = dir.y / sin_theta;
            alpha_2 = 1 / (std::pow(cos_phi / alpha_u_, 2) +
                           std::pow(sin_phi / alpha_v_, 2));
        }
        auto alpha_p = 2 / alpha_2 - 2;
        auto tan_theta_v_n = std::sqrt(1 - std::pow(cos_theta_v_n, 2)) / cos_theta_v_n;
        auto a = std::sqrt(std::max((Float)0, 0.5 * alpha_p + 1)) / tan_theta_v_n;

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