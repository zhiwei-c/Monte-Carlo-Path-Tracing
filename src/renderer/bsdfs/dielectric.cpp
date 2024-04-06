#include "csrt/renderer/bsdfs/dielectric.hpp"

#include "csrt/renderer/bsdfs/bsdf.hpp"
#include "csrt/renderer/bsdfs/kulla_conty.hpp"
#include "csrt/renderer/bsdfs/microfacet.hpp"
#include "csrt/rtcore/scene.hpp"
#include "csrt/utils.hpp"

namespace
{

using namespace csrt;

QUALIFIER_D_H float
EvaluateMultipleScatter(const DielectricData &data, const float N_dot_I,
                        const float N_dot_O, const float roughness,
                        const bool inside, const bool reflect)
{
    const float brdf_i = GetBrdfAvg(data.brdf_avg_buffer, N_dot_I, roughness),
                brdf_o = GetBrdfAvg(data.brdf_avg_buffer, N_dot_O, roughness),
                albedo_avg = GetAlbedoAvg(data.albedo_avg_buffer, roughness),
                f_ms = (1.0f - brdf_i) * (1.0f - brdf_o) /
                       (kPi * (1.0f - albedo_avg));

    const float F_avg = inside ? data.F_avg_inv : data.F_avg,
                eta = inside ? data.eta_inv : data.eta;

    const float f_add = pow(F_avg, 2) * albedo_avg /
                        (1.0f - F_avg * (1.0f - albedo_avg)),
                ratio_trans = ((1.0f - data.F_avg) * (1.0f - data.F_avg_inv) *
                               pow(eta, 2) /
                               ((1.0f - data.F_avg) +
                                (1.0f - data.F_avg_inv) * pow(eta, 2)));
    const float ret = f_ms * f_add * N_dot_I;
    return reflect ? (1.0f - ratio_trans) * ret : ratio_trans * ret;
}

} // namespace

namespace csrt
{


QUALIFIER_D_H void SampleDielectric(const DielectricData &data, uint32_t *seed,
                                    BsdfSampleRec *rec)
{
    const float scale = 1.2f - 0.2f * sqrt(abs(Dot(-rec->wo, rec->normal)));
    const float alpha_u = data.roughness_u->GetColor(rec->texcoord).x * scale,
                alpha_v = data.roughness_v->GetColor(rec->texcoord).x * scale;

    // 根据GGX法线分布函数重要抽样微平面法线
    Vec3 h_local(0);
    float D = 0;
    SampleGgx(RandomFloat(seed), RandomFloat(seed), alpha_u, alpha_v, &h_local,
              &D);
    const Vec3 h_world = rec->ToWorld(h_local);
    float H_dot_O = Dot(rec->wo, h_world);
    if (H_dot_O < kEpsilonFloat)
        return;

    float eta = data.eta;
    // 相对折射率的倒数，即入射侧介质和透射侧介质的绝对折射率之比
    float eta_inv = data.eta_inv;
    if (!rec->inside)
    { // 如果光线源于物体内部，那么应该颠倒相对折射率
        float temp = eta_inv;
        eta_inv = eta;
        eta = temp;
    }

    Vec3 wt;
    const bool full_reflect = !Ray::Refract(-rec->wo, h_world, eta, &wt);
    float F = FresnelSchlick(H_dot_O, data.reflectivity);
    const Vec3 wo_local = rec->ToLocal(rec->wo);
    if (full_reflect || RandomFloat(seed) < F)
    { // 抽样反射光线
        rec->wi = -Ray::Reflect(-rec->wo, h_world);
        const float N_dot_I = Dot(-rec->wi, rec->normal);
        if (N_dot_I < kEpsilonFloat)
            return;

        rec->pdf = F * D / (4.0f * H_dot_O);
        if (rec->pdf < kEpsilon)
            return;

        const Vec3 wi_local = rec->ToLocal(-rec->wi);
        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                    N_dot_O = wo_local.z;
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

        // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
        if (alpha_u == alpha_v)
        {
            rec->attenuation += EvaluateMultipleScatter(
                data, N_dot_I, N_dot_O, alpha_u, rec->inside, true);
        }

        const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    else
    { // 抽样折射光线
        rec->wi = -wt;
        Vec3 wi_local = rec->ToLocal(-rec->wi);
        wi_local.z = -wi_local.z;

        const float N_dot_I = wi_local.z;
        if (N_dot_I < kEpsilonFloat)
            return;

        const float H_dot_I = -Dot(wt, h_world);
        if (H_dot_I < kEpsilonFloat)
            return;

        H_dot_O = -H_dot_O;
        F = FresnelSchlick(H_dot_I, data.reflectivity);
        rec->pdf =
            ((1.0f - F) * D) * abs(H_dot_O / Sqr(eta_inv * H_dot_I + H_dot_O));
        if (rec->pdf < kEpsilon)
            return;

        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                    N_dot_O = wo_local.z;
        rec->attenuation =
            (((abs(H_dot_I) * abs(H_dot_O)) * ((1.0f - F) * G * D)) /
             abs(N_dot_O * Sqr(eta_inv * H_dot_I + H_dot_O)));

        // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
        if (alpha_u == alpha_v)
        {
            rec->attenuation += EvaluateMultipleScatter(
                data, N_dot_I, N_dot_O, alpha_u, !rec->inside, false);
        }

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);

        const Vec3 spec = data.specular_transmittance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    rec->valid = true;
}

QUALIFIER_D_H void EvaluateDielectric(const DielectricData &data,
                                      BsdfSampleRec *rec)
{
    float eta = data.eta;
    // 相对折射率的倒数，即入射侧介质和透射侧介质的绝对折射率之比
    float eta_inv = data.eta_inv;
    if (rec->inside)
    { // 如果光线源于物体内部，那么应该颠倒相对折射率
        float temp = eta_inv;
        eta_inv = eta;
        eta = temp;
    }

    // 计算微平面法线，使之与入射光线同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    const bool relfect = N_dot_O > 0.0f;
    const Vec3 h_world = relfect ? Normalize(-rec->wi + rec->wo)
                                 : -Normalize(eta_inv * (-rec->wi) + rec->wo),
               h_local = rec->ToLocal(h_world);

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const float alpha_u = data.roughness_u->GetColor(rec->texcoord).x,
                alpha_v = data.roughness_v->GetColor(rec->texcoord).x,
                D = PdfGgx(alpha_u, alpha_v, h_local),
                H_dot_I = Dot(-rec->wi, h_world),
                H_dot_O = Dot(rec->wo, h_world),
                F = FresnelSchlick(H_dot_I, data.reflectivity);
    rec->pdf = relfect ? (F * D) / (4.0f * H_dot_O)
                       : (((1.0f - F) * D) *
                          abs(H_dot_O / Sqr(eta_inv * H_dot_I + H_dot_O)));
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    const Vec3 wi_local = rec->ToLocal(-rec->wi);
    if (relfect)
    {
        const Vec3 wo_local = rec->ToLocal(rec->wo);
        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local);
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

        // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
        if (alpha_u == alpha_v)
        {
            const float N_dot_I = Dot(-rec->wi, rec->normal);
            rec->attenuation += EvaluateMultipleScatter(
                data, N_dot_I, N_dot_O, alpha_u, rec->inside, true);
        }

        const Vec3 spec = data.specular_reflectance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    else
    {
        const Vec3 wo_local = rec->ToLocal(-rec->wo);
        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local);
        rec->attenuation =
            (((abs(H_dot_I) * abs(H_dot_O)) * ((1.0f - F) * G * D)) /
             abs(N_dot_O * Sqr(eta_inv * H_dot_I + H_dot_O)));

        // 仅针对各向同性材料使用 Kulla-Conty 方法补偿损失的多次散射能量
        if (alpha_u == alpha_v)
        {
            const float N_dot_I = Dot(rec->normal, -rec->wi);
            rec->attenuation += EvaluateMultipleScatter(
                data, N_dot_I, N_dot_O, alpha_u, rec->inside, false);
        }

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);

        const Vec3 spec = data.specular_transmittance->GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
}

} // namespace csrt