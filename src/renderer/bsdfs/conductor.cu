#include "csrt/renderer/bsdfs/conductor.cuh"

#include "csrt/renderer/bsdfs/bsdf.cuh"
#include "csrt/renderer/bsdfs/kulla_conty.cuh"
#include "csrt/rtcore/scene.cuh"
#include "csrt/utils.cuh"

namespace
{

using namespace csrt;

QUALIFIER_D_H Vec3 EvaluateMultipleScatter(const ConductorData &data,
                                           const float N_dot_I,
                                           const float N_dot_O,
                                           const float roughness)
{
    const float brdf_i = GetBrdfAvg(data.brdf_avg_buffer, N_dot_I, roughness),
                brdf_o = GetBrdfAvg(data.brdf_avg_buffer, N_dot_O, roughness),
                albedo_avg = GetAlbedoAvg(data.albedo_avg_buffer, roughness),
                f_ms = (1.0f - brdf_i) * (1.0f - brdf_o) /
                       (kPi * (1.0f - albedo_avg));
    const Vec3 f_add = Sqr(data.F_avg) * albedo_avg /
                       (1.0f - data.F_avg * (1.0f - albedo_avg));
    return f_ms * f_add * N_dot_I;
}

} // namespace

namespace csrt
{

QUALIFIER_D_H void EvaluateConductor(const ConductorData &data,
                                     BsdfSampleRec *rec)
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilonFloat)
        return;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h_world = Normalize(-rec->wi + rec->wo),
               h_local = rec->ToLocal(h_world);
    const float alpha_u = data.roughness_u->GetColor(rec->texcoord).x,
                alpha_v = data.roughness_v->GetColor(rec->texcoord).x,
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
    const Vec3 F = FresnelSchlick(H_dot_I, data.reflectivity);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

    // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
    if (alpha_u == alpha_v)
    {
        const float N_dot_I = Dot(-rec->wi, rec->normal);
        const Vec3 compensation =
            EvaluateMultipleScatter(data, N_dot_I, N_dot_O, alpha_u);
        rec->attenuation += compensation;
    }

    const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

QUALIFIER_D_H void SampleConductor(const ConductorData &data, uint32_t *seed,
                                   BsdfSampleRec *rec)
{
    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h_local(0);
    float D = 0;
    const float alpha_u = data.roughness_u->GetColor(rec->texcoord).x,
                alpha_v = data.roughness_v->GetColor(rec->texcoord).x;
    SampleGgx(RandomFloat(seed), RandomFloat(seed), alpha_u, alpha_v, &h_local,
              &D);
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

    const Vec3 F = FresnelSchlick(H_dot_I, data.reflectivity);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

    // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
    if (alpha_u == alpha_v)
    {
        const Vec3 compensation =
            EvaluateMultipleScatter(data, N_dot_I, N_dot_O, alpha_u);
        rec->attenuation += compensation;
    }

    const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
    rec->attenuation *= spec;
}

} // namespace csrt