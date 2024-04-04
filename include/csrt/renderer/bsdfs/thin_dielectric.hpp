#ifndef CSRT__RENDERER__BSDF__THIN_DIELECTRIC_HPP
#define CSRT__RENDERER__BSDF__THIN_DIELECTRIC_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"
#include "dielectric.hpp"

namespace csrt
{

QUALIFIER_D_H void EvaluateThinDielectric(const DielectricData &data,
                                          BsdfSampleRec *rec);

QUALIFIER_D_H void SampleThinDielectric(const DielectricData &data,
                                        uint32_t *seed, BsdfSampleRec *rec);

} // namespace csrt

#endif