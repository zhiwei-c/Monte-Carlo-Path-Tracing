#ifndef CSRT__RENDERER__MEDIUM__ISOTROPIC_HPP
#define CSRT__RENDERER__MEDIUM__ISOTROPIC_HPP

#include "../../tensor.hpp"

namespace csrt
{

struct PhaseSampleRec;

QUALIFIER_D_H void EvaluateIsotropicPhase(PhaseSampleRec *rec);

QUALIFIER_D_H void SampleIsotropicPhase(uint32_t *seed, PhaseSampleRec *rec);

} // namespace csrt

#endif