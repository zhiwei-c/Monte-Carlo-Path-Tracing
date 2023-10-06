#include "conductor.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void Conductor::Evaluate(const float *pixel_buffer,
                                          Texture **texture_buffer,
                                          uint64_t *seed, SamplingRecord *rec) const
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilon)
        return;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h = Normalize(-rec->wi + rec->wo);
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x,
                D = PdfGgx(roughness, rec->normal, h),
                H_dot_O = Dot(rec->wo, h);
    rec->pdf = D / (4.0f * H_dot_O);
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                     SmithG1Ggx(roughness, rec->wo, rec->normal, h)),
                H_dot_I = Dot(-rec->wi, h),
                N_dot_I = Dot(-rec->wi, rec->normal);
    const Vec3 F = FresnelSchlick(H_dot_I, reflectivity_);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);
    rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness);
    const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
    rec->attenuation *= spec;
}

QUALIFIER_DEVICE void Conductor::Sample(const float *pixel_buffer,
                                        Texture **texture_buffer,
                                        uint64_t *seed, SamplingRecord *rec) const
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
    else
        rec->valid = true;

    const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                     SmithG1Ggx(roughness, rec->wo, rec->normal, h)),
                H_dot_I = H_dot_O,
                N_dot_O = Dot(rec->wo, rec->normal);
    const Vec3 F = FresnelSchlick(H_dot_I, reflectivity_);
    rec->attenuation = (F * D * G) / (4.0f * N_dot_O);
    rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness);
    const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
    rec->attenuation *= spec;
}

QUALIFIER_DEVICE Vec3 Conductor::EvaluateMultipleScatter(const float N_dot_I,
                                                         const float N_dot_O,
                                                         const float roughness) const
{
    const float brdf_i = GetBrdfAvg(N_dot_I, roughness),
                brdf_o = GetBrdfAvg(N_dot_O, roughness),
                albedo_avg = GetAlbedoAvg(roughness),
                f_ms = (1.0f - brdf_i) * (1.0f - brdf_o) / (kPi * (1.0f - albedo_avg));
    const Vec3 f_add = Sqr(F_avg_) * albedo_avg / (1.0f - F_avg_ * (1.0f - albedo_avg));
    return f_ms * f_add * N_dot_I;
}
