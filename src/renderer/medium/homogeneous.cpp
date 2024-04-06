#include "csrt/renderer/medium/homogeneous.hpp"

#include "csrt/renderer/medium/medium.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleHomogeneousMedium(const HomogeneousMediumData &data,
                                           const float max_distance,
                                           uint32_t *seed, MediumSampleRec *rec)
{
    float xi_0 = RandomFloat(seed);
    if (xi_0 < data.sampling_weight)
    { //抽样光线在介质内部是否发生散射
        xi_0 /= data.sampling_weight;
        const int channel = static_cast<int>(RandomFloat(seed) * 3);
        rec->distance = -log(1.0f - xi_0) / data.sigma_t[channel];
        if (rec->distance < max_distance)
        { //光线在介质内部发生了散射
            for (int dim = 0; dim < 3; ++dim)
            {
                rec->pdf +=
                    data.sigma_t[dim] * exp(-data.sigma_t[dim] * rec->distance);
            }
            rec->pdf *= data.sampling_weight * (1.0f / 3.0f);
            rec->scattered = true;
        }
    }

    if (!rec->scattered)
    { //光线在介质内部没有发生散射
        rec->distance = max_distance;
        rec->pdf = 0;
        for (int dim = 0; dim < 3; ++dim)
        {
            rec->pdf += exp(-data.sigma_t[dim] * rec->distance);
        }
        rec->pdf = data.sampling_weight * (1.0f / 3.0f) * rec->pdf +
                   (1.0f - data.sampling_weight);
    }

    for (int dim = 0; dim < 3; ++dim)
    {
        rec->attenuation[dim] = exp(-data.sigma_t[dim] * rec->distance);
        if (rec->attenuation[dim] > kEpsilonFloat)
            rec->valid = true;
    }
    if (rec->scattered)
        rec->attenuation *= data.sigma_s;
}

QUALIFIER_D_H void EvaluateHomogeneousMedium(const HomogeneousMediumData &data,
                                             MediumSampleRec *rec)
{
    for (int dim = 0; dim < 3; ++dim)
    {
        rec->attenuation[dim] = exp(-data.sigma_t[dim] * rec->distance);
        if (rec->attenuation[dim] > kEpsilonFloat)
            rec->valid = true;
    }

    if (!rec->valid)
        return;

    if (rec->scattered)
    {
        for (int dim = 0; dim < 3; ++dim)
            rec->pdf += data.sigma_t[dim] * rec->attenuation[dim];

        rec->pdf *= data.sampling_weight * (1.0f / 3.0f);
        rec->attenuation *= data.sigma_s;
    }
    else
    {
        for (int dim = 0; dim < 3; ++dim)
            rec->pdf += rec->attenuation[dim];

        rec->pdf = data.sampling_weight * (1.0f / 3.0f) * rec->pdf +
                   (1.0f - data.sampling_weight);
    }
}

} // namespace csrt