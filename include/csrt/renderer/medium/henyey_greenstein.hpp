#ifndef CSRT__RENDERER__MEDIUM__HENYEY_GREENSTEIN_HPP
#define CSRT__RENDERER__MEDIUM__HENYEY_GREENSTEIN_HPP

#include "../../tensor.hpp"

namespace csrt
{

struct PhaseSampleRec;

QUALIFIER_D_H void SampleHenyeyGreensteinPhase(const Vec3 &g, uint32_t *seed,
                                               PhaseSampleRec *rec);

QUALIFIER_D_H void EvaluateHenyeyGreensteinPhase(const Vec3 &g,
                                                 PhaseSampleRec *rec);

} // namespace csrt

#endif