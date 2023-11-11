#include "bsdf.cuh"

#include "rtcore.cuh"
#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H void rt::Bsdf::EvaluateConductor(SamplingRecord *rec) const
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilonFloat)
        return;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h_world = Normalize(-rec->wi + rec->wo),
               h_local = rec->ToLocal(h_world);
    const Texture &roughness_u =
                      data_.texture_buffer[data_.conductor.id_roughness_u],
                  &roughness_v =
                      data_.texture_buffer[data_.conductor.id_roughness_v];
    const float alpha_u = roughness_u.GetColor(rec->texcoord).x,
                alpha_v = roughness_v.GetColor(rec->texcoord).x,
                D = PdfGgx(alpha_u, alpha_v, h_local),
                H_dot_O = Dot(rec->wo, h_world);

    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilonFloat)
        return;
    else
        rec->valid = true;

    const Vec3 wi_local = rec->ToLocal(-rec->wi),
               wo_local = rec->ToLocal(rec->wo);
    const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                    SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                H_dot_I = Dot(-rec->wi, h_world);
    const Vec3 F = FresnelSchlick(H_dot_I, data_.conductor.reflectivity);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

    // const float N_dot_I = Dot(-rec->wi, rec->normal)
    // rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness);

    const Texture &specular_reflectance =
        data_.texture_buffer[data_.conductor.id_specular_reflectance];
    const Vec3 spec = specular_reflectance.GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

QUALIFIER_D_H void rt::Bsdf::SampleConductor(const Vec3 &xi,
                                             SamplingRecord *rec) const
{
    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h_local(0);
    float D = 0;
    const Texture &roughness_u =
                      data_.texture_buffer[data_.conductor.id_roughness_u],
                  &roughness_v =
                      data_.texture_buffer[data_.conductor.id_roughness_v];
    const float alpha_u = roughness_u.GetColor(rec->texcoord).x,
                alpha_v = roughness_v.GetColor(rec->texcoord).x;
    SampleGgx(xi.x, xi.y, alpha_u, alpha_v, &h_local, &D);
    const Vec3 h_world = rec->ToWorld(h_local);
    
    const float H_dot_O = Dot(rec->wo, h_world);
    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilonFloat)
        return;

    rec->wi = -Ray::Reflect(-rec->wo, h_world);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    if (N_dot_I < kEpsilonFloat)
        return;
    else
        rec->valid = true;

    const Vec3 wi_local = rec->ToLocal(-rec->wi),
               wo_local = rec->ToLocal(rec->wo);
    const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                    SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                H_dot_I = Dot(-rec->wi, h_world), N_dot_O = wo_local.z;

    const Vec3 F = FresnelSchlick(H_dot_I, data_.conductor.reflectivity);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

    // const float N_dot_I = Dot(-rec->wi, rec->normal)
    // rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness);

    const Texture &specular_reflectance =
        data_.texture_buffer[data_.conductor.id_specular_reflectance];
    const Vec3 spec = specular_reflectance.GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

} // namespace rt