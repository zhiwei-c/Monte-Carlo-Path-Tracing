#include "phase_function.hpp"

#include "../../math/coordinate.hpp"
#include "../../math/sample.hpp"
#include "../../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

void HenyeyGreensteinPhaseFunction::Sample(SamplingRecord *rec, Sampler *sampler) const
{
    auto channel = std::min(static_cast<int>(sampler->Next1D() * 3), 2);
    double g = g_[channel];

    double cos_theta = 0;
    if (std::abs(g) < kEpsilonCompare)
    {
        cos_theta = 1.0 - 2.0 * sampler->Next1D();
    }
    else
    {
        double sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * sampler->Next1D());
        cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    }

    rec->pdf = 0;
    for (int dim = 0; dim < 3; ++dim)
    {
        double temp = 1.0 + g_[dim] * g_[dim] + 2.0 * g_[dim] * cos_theta;
        rec->attenuation[dim] = (0.25 * kPiRcp) * (1.0 - g_[dim] * g_[dim]) / (temp * std::sqrt(temp));
        rec->pdf += rec->attenuation[dim];
    }
    rec->pdf *= (1.0 / 3.0);
    if (rec->pdf <= 0.0)
    {
        return;
    }

    rec->type = ScatteringType::kScattering;
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double phi = 2 * kPi * sampler->Next1D();
    rec->wi = dvec3(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
    rec->wi = -ToWorld(rec->wi, rec->wo);
}

void HenyeyGreensteinPhaseFunction::Eval(SamplingRecord *rec) const
{
    double cos_theta = glm::dot(-rec->wi, rec->wo);
    for (int dim = 0; dim < 3; ++dim)
    {
        double temp = 1.0 + g_[dim] * g_[dim] + 2.0 * g_[dim] * cos_theta;
        rec->attenuation[dim] = (0.25 * kPiRcp) * (1.0 - g_[dim] * g_[dim]) / (temp * std::sqrt(temp));
        rec->pdf += rec->attenuation[dim];
    }
    rec->pdf *= (1.0 / 3.0);
    if (rec->pdf <= 0.0)
    {
        return;
    }

    rec->type = ScatteringType::kScattering;
}

NAMESPACE_END(raytracer)