#include "bsdf.cuh"

#include "utils.cuh"

namespace csrt
{

QUALIFIER_D_H void Bsdf::EvaluateDiffuse(SamplingRecord *rec) const
{
    // 反射光线与法线方向应该位于同侧
    rec->pdf = Dot(rec->wo, rec->normal);
    if (rec->pdf < kEpsilonFloat)
        return;
    rec->valid = true;

    const Texture &diffuse_reflectance =
        data_.texture_buffer[data_.diffuse.id_diffuse_reflectance];
    const Vec3 albedo = diffuse_reflectance.GetColor(rec->texcoord);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

QUALIFIER_D_H void Bsdf::SampleDiffuse(const Vec3 &xi,
                                       SamplingRecord *rec) const
{
    Vec3 wi_local;
    SampleHemisCos(xi.x, xi.y, &wi_local, &rec->pdf);
    if (rec->pdf < kEpsilonFloat)
        return;
    rec->wi = -rec->ToWorld(wi_local);
    rec->valid = true;
    const Texture &diffuse_reflectance =
        data_.texture_buffer[data_.diffuse.id_diffuse_reflectance];
    const Vec3 albedo = diffuse_reflectance.GetColor(rec->texcoord);
    const float N_dot_I = wi_local.z;
    rec->attenuation = albedo * k1DivPi * N_dot_I;
}

} // namespace csrt