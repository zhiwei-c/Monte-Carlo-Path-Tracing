#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"
#include "dielectric.cuh"

namespace csrt
{

QUALIFIER_D_H void EvaluateThinDielectric(const DielectricData &data,
                                          BsdfSampleRec *rec);

QUALIFIER_D_H void SampleThinDielectric(const DielectricData &data,
                                        uint32_t *seed, BsdfSampleRec *rec);

} // namespace csrt