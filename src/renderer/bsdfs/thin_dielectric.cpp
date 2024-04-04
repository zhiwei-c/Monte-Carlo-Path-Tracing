#include "csrt/renderer/bsdfs/thin_dielectric.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/renderer/bsdfs/microfacet.hpp"
#include "csrt/rtcore/scene.hpp"
#include "csrt/utils.hpp"

namespace csrt
{

QUALIFIER_D_H void EvaluateThinDielectric(const DielectricData &data,
                                          BsdfSampleRec *rec)
{

    bool reflect = true;
    Vec3 wo = rec->wo;
    float N_dot_O = Dot(rec->wo, rec->normal);
    if (fabs(N_dot_O) < kEpsilonFloat)
        return;

    // 调整反射光线方向，使之与法线方向位于同侧
    Vec3 wo_local = rec->ToLocal(rec->wo);
    if (N_dot_O < 0.0f)
    {
        reflect = false;
        N_dot_O = -N_dot_O;
        wo_local.z = -wo_local.z;
        wo = rec->ToWorld(wo_local);
    }

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h_world = Normalize(-rec->wi + wo),
               h_local = rec->ToLocal(h_world);
    const float alpha_u = data.roughness_u->GetColor(rec->texcoord).x,
                alpha_v = data.roughness_v->GetColor(rec->texcoord).x,
                D = PdfGgx(alpha_u, alpha_v, h_local),
                H_dot_I = Dot(-rec->wi, h_world),
                H_dot_O = Dot(rec->wo, h_world);
    float F = FresnelSchlick(H_dot_I, data.reflectivity);
    if (F < 1.0f)
        F *= 2.0f / (1.0f + F);

    rec->pdf = reflect ? (F * D) / (4.0f * H_dot_O)
                       : ((1.0f - F) * D) / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    const Vec3 wi_local = rec->ToLocal(-rec->wi);
    const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                    SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local);
    if (reflect)
    {
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    else
    {
        rec->attenuation = ((1.0f - F) * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = data.specular_transmittance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
}

QUALIFIER_D_H void SampleThinDielectric(const DielectricData &data,
                                        uint32_t *seed, BsdfSampleRec *rec)
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

    const Vec3 wi_local = rec->ToLocal(-rec->wi),
               wo_local = rec->ToLocal(rec->wo);
    const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                    SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                H_dot_I = Dot(-rec->wi, h_world), N_dot_O = wo_local.z;

    float F = FresnelSchlick(H_dot_I, data.reflectivity);
    if (F < 1.0f)
        F *= 2.0f / (1.0f + F);

    if (RandomFloat(seed) < F)
    {
        rec->pdf *= F;
        if (rec->pdf < kEpsilon)
            return;

        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

        const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    else
    {
        rec->pdf *= 1.0f - F;
        if (rec->pdf < kEpsilon)
            return;

        rec->attenuation = ((1.0f - F) * D * G) / (4.0f * N_dot_O);

        const Vec3 spec = data.specular_transmittance->GetColor(rec->texcoord);
        rec->attenuation *= spec;

        rec->wi = rec->wo;
    }

    rec->valid = true;
}

} // namespace csrt