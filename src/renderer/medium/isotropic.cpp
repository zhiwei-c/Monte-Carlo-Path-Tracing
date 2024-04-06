#include "csrt/renderer/medium/isotropic.hpp"

#include "csrt/renderer/medium/medium.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleIsotropicPhase(uint32_t *seed, PhaseSampleRec *rec)
{
    rec->valid = true;
    rec->attenuation = Vec3(k1Div4Pi);
    rec->pdf = k1Div4Pi;
    rec->wi = SampleSphereUniform(RandomFloat(seed), RandomFloat(seed));
}

QUALIFIER_D_H void EvaluateIsotropicPhase(PhaseSampleRec *rec)
{
    rec->valid = true;
    rec->attenuation = Vec3(k1Div4Pi);
    rec->pdf = k1Div4Pi;
}

} // namespace csrt