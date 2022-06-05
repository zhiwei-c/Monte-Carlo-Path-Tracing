#pragma once

#include "../core/phase_function_base.h"

NAMESPACE_BEGIN(raytracer)

class HenyeyGreensteinPhase : public PhaseFunction
{
public:
    HenyeyGreensteinPhase(const Spectrum &g) : g_(g) {}

    void Sample(SamplingRecord &rec) const override
    {
        auto channel = std::min(static_cast<int>(UniformFloat2() * 3), 2);
        Float g = g_[channel];

        Float cos_theta = 0;
        if (std::abs(g) < kEpsilon)
        {
            cos_theta = 1.0 - 2.0 * UniformFloat();
        }
        else
        {
            Float sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * UniformFloat());
            cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
        }
        Float sin_theta = std::sqrt(std::max(kEpsilon, 1.0 - cos_theta * cos_theta));
        Float phi = 2 * kPi * UniformFloat();

        rec.wi = Vector3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
        rec.wi = -ToWorld(rec.wi, rec.wo);

        rec.pdf = 0;
        for (int i = 0; i < 3; i++)
        {
            Float temp = 1.0 + g_[i] * g_[i] + 2.0 * g_[i] * cos_theta;
            rec.attenuation[i] = kFourPiInv * (1.0 - g_[i] * g_[i]) / (temp * std::sqrt(temp));
            rec.pdf += rec.attenuation[i];
        }
        rec.pdf /= 3.0;
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kScattering;
    }

    void Eval(SamplingRecord &rec) const override
    {
        auto attenuation = Spectrum(1);

        Float cos_theta = glm::dot(-rec.wi, rec.wo);
        for (int i = 0; i < 3; i++)
        {
            Float temp = 1.0 + g_[i] * g_[i] + 2.0 * g_[i] * cos_theta;
            rec.attenuation[i] = kFourPiInv * (1.0 - g_[i] * g_[i]) / (temp * std::sqrt(temp));
            rec.pdf += rec.attenuation[i];
        }
        rec.pdf /= 3.0;
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kScattering;
    }

private:
    Spectrum g_;
};

NAMESPACE_END(raytracer)