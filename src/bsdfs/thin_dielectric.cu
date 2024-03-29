#include "thin_dielectric.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void ThinDielectric::Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                               uint32_t *seed, SamplingRecord *rec) const
{
    bool reflect = true;
    Vec3 wo = rec->wo;
    // 反射光线与法线方向应该位于同侧
    float N_dot_O = Dot(rec->wo, rec->normal);
    if (fabs(N_dot_O) < kEpsilonFloat)
    {
        return;
    }
    else if (N_dot_O < 0.0f)
    {
        reflect = false;
        N_dot_O = -N_dot_O;
        wo = ToWorld(ToLocal(wo, -rec->normal), rec->normal);
    }

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h = Normalize(-rec->wi + wo);
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x,
                D = PdfGgx(roughness, rec->normal, h),
                H_dot_O = Dot(wo, h),
                H_dot_I = Dot(-rec->wi, h);
    float F = FresnelSchlick(H_dot_I, reflectivity_);
    if (F < 1.0f)
        F *= 2.0f / (1.0f + F);
    if (reflect)
    {
        rec->pdf = (F * D) / (4.0f * H_dot_O);
        if (rec->pdf < kEpsilon)
            return;
        else
            rec->valid = true;

        const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, wo, rec->normal, h));
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord,
                                                                             pixel_buffer);
        rec->attenuation *= spec;
    }
    else
    {
        rec->pdf = ((1.0f - F) * D) / (4.0f * H_dot_O);
        if (rec->pdf < kEpsilon)
            return;
        else
            rec->valid = true;

        const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, wo, rec->normal, h));
        rec->attenuation = ((1.0f - F) * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = texture_buffer[id_specular_transmittance_]->GetColor(rec->texcoord,
                                                                               pixel_buffer);
        rec->attenuation *= spec;
    }
}

QUALIFIER_DEVICE void ThinDielectric::Sample(Texture **texture_buffer, const float *pixel_buffer,
                                             uint32_t *seed, SamplingRecord *rec) const
{
    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h(0);
    float D = 0;
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x;
    SampleGgx(RandomFloat(seed), RandomFloat(seed), roughness, h, D);
    h = ToWorld(h, rec->normal);

    const float H_dot_O = Dot(rec->wo, h);
    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilon)
        return;

    rec->wi = -Reflect(-rec->wo, h);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    if (N_dot_I < kEpsilon)
        return;

    const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                     SmithG1Ggx(roughness, rec->wo, rec->normal, h)),
                H_dot_I = H_dot_O,
                N_dot_O = Dot(rec->wo, rec->normal);
    float F = FresnelSchlick(H_dot_I, reflectivity_);
    if (F < 1.0f)
        F *= 2.0f / (1.0f + F);

    if (RandomFloat(seed) < F)
    {
        rec->pdf *= F;
        if (rec->pdf < kEpsilon)
            return;
        else
            rec->valid = true;

        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord,
                                                                             pixel_buffer);
        rec->attenuation *= spec;
    }
    else
    {
        rec->pdf *= 1.0f - F;
        if (rec->pdf < kEpsilon)
            return;
        else
            rec->valid = true;

        rec->attenuation = ((1.0f - F) * D * G) / (4.0f * N_dot_O);
        const Vec3 spec = texture_buffer[id_specular_transmittance_]->GetColor(rec->texcoord,
                                                                               pixel_buffer);
        rec->attenuation *= spec;

        rec->wi = rec->wo;
    }
}
