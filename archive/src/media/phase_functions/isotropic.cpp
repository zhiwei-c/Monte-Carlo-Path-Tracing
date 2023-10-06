#include "phase_function.hpp"

#include "../../math/sample.hpp"
#include "../../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

void IsotropicPhaseFunction::Sample(SamplingRecord *rec, Sampler *sampler) const
{
    rec->wi = SampleSphereUniform(sampler->Next2D());
    rec->attenuation = dvec3(0.25 * kPiRcp);
    rec->pdf = 0.25 * kPiRcp;
    rec->type = ScatteringType::kScattering;
}

void IsotropicPhaseFunction::Eval(SamplingRecord *rec) const
{
    rec->attenuation = dvec3(0.25 * kPiRcp);
    rec->pdf = 0.25 * kPiRcp;
    rec->type = ScatteringType::kScattering;
}

NAMESPACE_END(raytracer)