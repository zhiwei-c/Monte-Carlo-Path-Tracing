#include "csrt/renderer/bsdfs/diffuse.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void EvaluateDiffuse(const DiffuseData &data, BsdfSampleRec *rec)
{
    // 反射光线与法线方向应该位于同侧
    rec->pdf = Dot(rec->wo, rec->normal);
    if (rec->pdf < kEpsilon)
        return;
    rec->valid = true;

    const Vec3 albedo = data.diffuse_reflectance->GetColor(rec->texcoord);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

QUALIFIER_D_H void SampleDiffuse(const DiffuseData &data, uint32_t *seed,
                                 BsdfSampleRec *rec)
{
    Vec3 wi_local;
    SampleHemisCos(RandomFloat(seed), RandomFloat(seed), &wi_local, &rec->pdf);
    if (rec->pdf < kEpsilon)
        return;
    rec->wi = -rec->ToWorld(wi_local);
    rec->valid = true;
    const Vec3 albedo = data.diffuse_reflectance->GetColor(rec->texcoord);
    const float N_dot_I = wi_local.z;
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

} // namespace csrt