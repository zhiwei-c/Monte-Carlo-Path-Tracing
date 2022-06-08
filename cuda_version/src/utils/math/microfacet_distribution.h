#pragma once

#include <utility>

#include "../global.h"

#include "coordinate.h"

enum MicrofacetDistribType
{
    kNoneDistrib,
    kBeckmann,
    kGgx
};

__device__ inline void SampleNormDistrib(MicrofacetDistribType type, Float alpha_u, Float alpha_v, const vec3 &macro_normal,
                                         const vec3 &sample, vec3 &facet_normal, Float &pdf)
{
    bool isotropic = (alpha_u == alpha_v);
    Float sin_phi = 0,
         cos_phi = 0,
         sin_theta = 0,
         cos_theta = 0,
         alpha_2 = 0;
    switch (type)
    {
    case kGgx:
    {
        if (isotropic)
        {
            auto phi = 2.0 * kPi * sample.y;
            cos_phi = cos(phi);
            sin_phi = sin(phi);
            alpha_2 = alpha_u * alpha_u;
        }
        else
        {
            auto phi = atan(alpha_v / alpha_u * tan(kPi + 2.0 * kPi * sample.y)) + kPi * static_cast<int>(2.0 * sample.y + 0.5);
            cos_phi = cos(phi);
            sin_phi = sin(phi);
            alpha_2 = 1.0 / (pow(cos_phi / alpha_u, 2) + pow(sin_phi / alpha_v, 2));
        }
        auto tan_theta_2 = alpha_2 * sample.x / (1.0 - sample.x);
        cos_theta = 1.0 / sqrt(1.0 + tan_theta_2);
        sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        pdf = 1.0 / (kPi * alpha_u * alpha_v * pow(cos_theta, 3) * pow(1.0 + tan_theta_2 / alpha_2, 2));
        break;
    }
    default:
    {
        if (isotropic)
        {
            auto phi = 2.0 * kPi * sample.y;
            cos_phi = cos(phi);
            sin_phi = sin(phi);
            alpha_2 = alpha_u * alpha_u;
        }
        else
        {
            auto phi = atan(alpha_v / alpha_u * tan(kPi + 2.0 * kPi * sample.y)) + kPi * static_cast<int>(2 * sample.y + 0.5);
            cos_phi = cos(phi);
            sin_phi = sin(phi);
            alpha_2 = 1.0 / (pow(cos_phi / alpha_u, 2) +
                             pow(sin_phi / alpha_v, 2));
        }
        cos_theta = 1.0 / sqrt(1.0 - alpha_2 * log(1.0 - sample.x));
        sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        pdf = (1.0 - sample.x) / (kPiInv * alpha_u * alpha_v * pow(cos_theta, 3));
        break;
    }
    }
    facet_normal = vec3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    facet_normal = ToWorld(facet_normal, macro_normal);
}

__device__ inline Float PdfNormDistrib(MicrofacetDistribType type, Float alpha_u, Float alpha_v, const vec3 &macro_normal,
                                       const vec3 &facet_normal)
{
    auto cos_theta = myvec::dot(macro_normal, facet_normal);
    if (cos_theta <= 0)
        return 0;

    auto cos_theta_2 = cos_theta * cos_theta,
         tan_theta_2 = (1.0 - cos_theta_2) / cos_theta_2,
         cos_theta_3 = pow(cos_theta, 3);
    auto alpha_2 = alpha_u * alpha_v;

    auto isotropic = (alpha_u == alpha_v);
    switch (type)
    {
    case kGgx:
    {
        if (isotropic)
            return alpha_2 / (kPi * cos_theta_3 * pow(alpha_2 + tan_theta_2, 2));
        else
        {
            auto dir = ToLocal(facet_normal, macro_normal);
            return cos_theta / (kPi * alpha_2 * pow(pow(dir.x / alpha_u, 2) + pow(dir.y / alpha_v, 2) + pow(dir.z, 2), 2));
        }
        break;
    }
    default:
    {
        if (isotropic)
            return exp(-tan_theta_2 / alpha_2) / (kPi * alpha_2 * cos_theta_3);
        else
        {
            auto dir = ToLocal(facet_normal, macro_normal);
            return exp(-(pow(dir.x / alpha_u, 2) + pow(dir.y / alpha_v, 2)) / cos_theta_2) / (kPi * alpha_2 * cos_theta_3);
        }
        break;
    }
    }
    return 0;
}

__device__ inline Float SmithG1(MicrofacetDistribType type, Float alpha_u, Float alpha_v, const vec3 &v, const vec3 &macro_normal,
                                const vec3 &facet_normal)
{
    auto cos_v_n = myvec::dot(v, macro_normal);
    auto cos_v_m = myvec::dot(v, facet_normal);
    if (cos_v_n * cos_v_m <= 0)
        return 0;

    if (abs(cos_v_n - 1) < kEpsilon)
        return 1;

    auto isotropic = (alpha_u == alpha_v);
    switch (type)
    {
    case kGgx:
    {
        if (isotropic)
        {
            auto cos_v_n_2 = cos_v_n * cos_v_n;
            auto tan_v_n_2 = (1.0 - cos_v_n_2) / cos_v_n_2;
            auto alpha_2 = alpha_u * alpha_u;
            return 2.0 / (1.0 + sqrt(1.0 + alpha_2 * tan_v_n_2));
        }
        else
        {
            auto dir = ToLocal(v, macro_normal);
            Float xy_alpha_2 = pow(alpha_u * dir.x, 2) + pow(alpha_v * dir.y, 2),
                  tan_v_n_alpha_2 = xy_alpha_2 / pow(dir.z, 2);
            return 2.0 / (1.0 + sqrt(1.0 + tan_v_n_alpha_2));
        }
        break;
    }
    default:
    {
        auto a = static_cast<Float>(0);
        if (isotropic)
        {
            auto tan_theta_v_n = sqrt(1.0 - pow(cos_v_n, 2)) / cos_v_n;
            a = 1.0 / (alpha_u * tan_theta_v_n);
        }
        else
        {
            auto dir = ToLocal(v, macro_normal);
            auto xy_alpha_2 = pow(alpha_u * dir.x, 2) + pow(alpha_v * dir.y, 2);
            auto tan_theta_alpha_2 = xy_alpha_2 / pow(dir.z, 2);
            a = 1.0 / sqrt(tan_theta_alpha_2);
        }

        if (a < 1.6)
        {
            auto a_2 = a * a;
            return (3.535 * a + 2.181 * a_2) / (1.0 + 2.276 * a + 2.577 * a_2);
        }
        else
            return 1;
        break;
    }
    }
    return 0;
}
