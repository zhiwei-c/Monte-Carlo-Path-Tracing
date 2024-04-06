#ifndef CSRT__RENDERER__MEDIUM__HOMOGENEOUS_HPP
#define CSRT__RENDERER__MEDIUM__HOMOGENEOUS_HPP

#include "../../tensor.hpp"

namespace csrt
{

struct MediumSampleRec;

struct HomogeneousMediumInfo
{
    Vec3 sigma_a = {};
    Vec3 sigma_s = {};
};

struct HomogeneousMediumData
{
    float sampling_weight = 0.0f;
    Vec3 sigma_s = {};
    Vec3 sigma_t = {};
};

QUALIFIER_D_H void SampleHomogeneousMedium(const HomogeneousMediumData &data,
                                           const float max_distance,
                                           uint32_t *seed,
                                           MediumSampleRec *rec);

QUALIFIER_D_H void EvaluateHomogeneousMedium(const HomogeneousMediumData &data,
                                             MediumSampleRec *rec);

} // namespace csrt

#endif