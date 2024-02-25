#include "csrt/renderer/bsdf.cuh"

#include "csrt/utils.cuh"


namespace csrt
{

QUALIFIER_D_H void BSDF::EvaluateDiffuse(BSDF::SampleRec *rec) const
{
    // 反射光线与法线方向应该位于同侧
    rec->pdf = Dot(rec->wo, rec->normal);
    if (rec->pdf < kEpsilon)
        return;
    rec->valid = true;

    const Vec3 albedo =
        data_.diffuse.diffuse_reflectance->GetColor(rec->texcoord);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

QUALIFIER_D_H void BSDF::SampleDiffuse(uint32_t *seed,
                                       BSDF::SampleRec *rec) const
{
    Vec3 wi_local;
    SampleHemisCos(RandomFloat(seed), RandomFloat(seed), &wi_local, &rec->pdf);
    if (rec->pdf < kEpsilon)
        return;
    rec->wi = -rec->ToWorld(wi_local);
    rec->valid = true;
    const Vec3 albedo =
        data_.diffuse.diffuse_reflectance->GetColor(rec->texcoord);
    const float N_dot_I = wi_local.z;
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

} // namespace csrt