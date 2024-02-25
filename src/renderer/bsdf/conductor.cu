#include "csrt/renderer/bsdf.cuh"

#include "csrt/rtcore.cuh"
#include "csrt/utils.cuh"

namespace csrt
{

QUALIFIER_D_H void BSDF::EvaluateConductor(BSDF::SampleRec *rec) const
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilonFloat)
        return;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h_world = Normalize(-rec->wi + rec->wo),
               h_local = rec->ToLocal(h_world);
    const float alpha_u =
                    data_.conductor.roughness_u->GetColor(rec->texcoord).x,
                alpha_v =
                    data_.conductor.roughness_v->GetColor(rec->texcoord).x,
                D = PdfGgx(alpha_u, alpha_v, h_local),
                H_dot_O = Dot(rec->wo, h_world);

    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilon)
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

    const Vec3 spec =
        data_.conductor.specular_reflectance->GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

QUALIFIER_D_H void BSDF::SampleConductor(uint32_t *seed,
                                         BSDF::SampleRec *rec) const
{
    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h_local(0);
    float D = 0;
    const float alpha_u =
                    data_.conductor.roughness_u->GetColor(rec->texcoord).x,
                alpha_v =
                    data_.conductor.roughness_v->GetColor(rec->texcoord).x;
    SampleGgx(RandomFloat(seed), RandomFloat(seed), alpha_u, alpha_v, &h_local, &D);
    const Vec3 h_world = rec->ToWorld(h_local);

    const float H_dot_O = Dot(rec->wo, h_world);
    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilon)
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

    const Vec3 spec =
        data_.conductor.specular_reflectance->GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

} // namespace csrt