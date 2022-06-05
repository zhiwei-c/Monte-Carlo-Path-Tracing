#pragma once

#include "../core/phase_function_base.h"

NAMESPACE_BEGIN(raytracer)

class IsotropicPhase : public PhaseFunction
{
public:
    IsotropicPhase() {}

    void Sample(SamplingRecord &rec) const override
    {
        rec.wi = SphereUniform();
        rec.attenuation = Spectrum(kFourPiInv);
        rec.pdf = kFourPiInv;
        rec.type = ScatteringType::kScattering;
    }

    void Eval(SamplingRecord &rec) const override
    {
        rec.attenuation = Spectrum(kFourPiInv);
        rec.pdf = kFourPiInv;
        rec.type = ScatteringType::kScattering;
    }
};

NAMESPACE_END(raytracer)