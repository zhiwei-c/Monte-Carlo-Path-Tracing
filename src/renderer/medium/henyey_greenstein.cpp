#include "csrt/renderer/medium/henyey_greenstein.hpp"

#include "csrt/renderer/medium/medium.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleHenyeyGreensteinPhase(const Vec3 &g, uint32_t *seed,
                                               PhaseSampleRec *rec)
{
    const int channel = static_cast<int>(RandomFloat(seed) * 3);

    float cos_theta = 0;
    if (abs(g[channel]) < kEpsilonFloat)
    {
        cos_theta = 1.0f - 2.0f * RandomFloat(seed);
    }
    else
    {
        const float sqr_term =
            (1.0f - Sqr(g[channel])) /
            (1.0f - g[channel] + 2.0f * g[channel] * RandomFloat(seed));
        cos_theta =
            (1.0f + Sqr(g[channel]) - Sqr(sqr_term)) / (2.0f * g[channel]);
    }

    const Vec3 temp = 1.0f + Sqr(g) + 2.0f * cos_theta * g;
    rec->attenuation = k1Div4Pi * (1.0f - Sqr(g)) / (temp * Sqrt(temp));

    rec->pdf = 0;
    for (int dim = 0; dim < 3; ++dim)
        rec->pdf += rec->attenuation[dim];
    rec->pdf *= (1.0f / 3.0f);
    if (rec->pdf < kEpsilon)
        return;

    rec->valid = true;
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - Sqr(cos_theta)));
    const float phi = k2Pi * RandomFloat(seed);
    rec->wi = {sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta};
    rec->wi = -LocalToWorld(rec->wi, rec->wo);
}

QUALIFIER_D_H void EvaluateHenyeyGreensteinPhase(const Vec3 &g,
                                                 PhaseSampleRec *rec)
{
    const float cos_theta = Dot(-rec->wi, rec->wo);

    const Vec3 temp = 1.0f + Sqr(g) + 2.0f * cos_theta * g;
    rec->attenuation = k1Div4Pi * (1.0f - Sqr(g)) / (temp * Sqrt(temp));

    rec->pdf = 0;
    for (int dim = 0; dim < 3; ++dim)
        rec->pdf += rec->attenuation[dim];
    rec->pdf *= (1.0f / 3.0f);
    if (rec->pdf < kEpsilon)
        return;

    rec->valid = true;
}

} // namespace csrt