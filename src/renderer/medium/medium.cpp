#include "csrt/renderer/medium/medium.hpp"

namespace csrt
{

QUALIFIER_D_H Medium::Medium(const uint32_t id, const MediumInfo &info)
    : id_(id)
{
    data_.type = info.type;
    data_.phase_func = info.phase_func;
    switch (data_.type)
    {
    case MediumType::kHomogeneous:
    {
        data_.homogeneous.sigma_s = info.homogeneous.sigma_s;
        data_.homogeneous.sigma_t =
            info.homogeneous.sigma_a + info.homogeneous.sigma_s;
        const Vec3 albedo =
            info.homogeneous.sigma_s /
            (info.homogeneous.sigma_a + info.homogeneous.sigma_s);
        for (int dim = 0; dim < 3; ++dim)
        {
            if (albedo[dim] > data_.homogeneous.sampling_weight &&
                data_.homogeneous.sigma_t[dim] > 0)
            {
                data_.homogeneous.sampling_weight = albedo[dim];
            }
        }
        if (data_.homogeneous.sampling_weight > 0)
        {
            if (data_.homogeneous.sampling_weight < 0.5f)
                data_.homogeneous.sampling_weight = 0.5f;
        }
        break;
    }
    default:
        break;
    }
}

QUALIFIER_D_H void Medium::Sample(const float max_distance, uint32_t *seed,
                                  MediumSampleRec *rec) const
{
    switch (data_.type)
    {
    case MediumType::kHomogeneous:
        SampleHomogeneousMedium(data_.homogeneous, max_distance, seed, rec);
        break;
    }
}

QUALIFIER_D_H void Medium::Evaluate(MediumSampleRec *rec) const
{
    switch (data_.type)
    {
    case MediumType::kHomogeneous:
        EvaluateHomogeneousMedium(data_.homogeneous, rec);
        break;
    }
}

QUALIFIER_D_H void Medium::EvaluatePhase(PhaseSampleRec *rec) const
{
    switch (data_.phase_func.type)
    {
    case PhaseFunctionType::kIsotropic:
        EvaluateIsotropicPhase(rec);
        break;
    case PhaseFunctionType::kHenyeyGreenstein:
        EvaluateHenyeyGreensteinPhase(data_.phase_func.g, rec);
        break;
    }
}

QUALIFIER_D_H void Medium::SamplePhase(uint32_t *seed,
                                       PhaseSampleRec *rec) const
{
    switch (data_.phase_func.type)
    {
    case PhaseFunctionType::kIsotropic:
        SampleIsotropicPhase(seed, rec);
        break;
    case PhaseFunctionType::kHenyeyGreenstein:
        SampleHenyeyGreensteinPhase(data_.phase_func.g, seed, rec);
        break;
    }
}

} // namespace csrt