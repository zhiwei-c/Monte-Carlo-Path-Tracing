#include "plastic.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void Plastic::Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                        uint32_t *seed, SamplingRecord *rec) const
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilon)
        return;

    // 计算塑料清漆层和基底层反射的权重
    const Vec3 kd = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord, pixel_buffer),
               ks = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
    float weight_spec = (ks.x + ks.y + ks.z) / ((kd.x + kd.y + kd.z) + (ks.x + ks.y + ks.z));
    const float N_dot_I = Dot(-rec->wi, rec->normal),
                kr_i = FresnelSchlick(N_dot_I, reflectivity_);
    float pdf_spec = kr_i * weight_spec,
          pdf_diff = (1.0f - kr_i) * (1.0f - weight_spec);
    pdf_spec = pdf_spec / (pdf_spec + pdf_diff);
    pdf_diff = 1.0f - pdf_spec;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h = Normalize(-rec->wi + rec->wo);
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x,
                D = PdfGgx(roughness, rec->normal, h),
                H_dot_O = Dot(rec->wo, h);
    pdf_spec *= D / (4.0f * H_dot_O);

    // 反推余弦加权重要抽样时的概率
    pdf_diff *= PdfHemisCos(ToLocal(rec->wo, rec->normal));

    rec->pdf = pdf_spec + pdf_diff;
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    // 计算塑料清漆层贡献的光能衰减系数
    if (pdf_spec >= kEpsilon)
    {
        const float H_dot_I = Dot(-rec->wi, h),
                    F = FresnelSchlick(H_dot_I, reflectivity_),
                    G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, rec->wo, rec->normal, h));
        Vec3 spec = (F * D * G) / (4.0f * N_dot_O);
        spec *= texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
        rec->attenuation += spec;
    }

    // 计算塑料基底层贡献的光能衰减系数
    if (pdf_diff >= kEpsilon)
    {
        const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                              pixel_buffer);
        Vec3 diff = albedo * kPiInv * N_dot_I;
        const float kr_o = FresnelSchlick(N_dot_O, reflectivity_);
        diff *= ((1.0f - kr_i) * (1.0f - kr_o)) / (1.0f - F_avg_);
        rec->attenuation += diff;
    }
}

QUALIFIER_DEVICE void Plastic::Sample(Texture **texture_buffer, const float *pixel_buffer,
                                      uint32_t *seed, SamplingRecord *rec) const
{
    // 计算塑料清漆层和基底层反射的权重
    const Vec3 kd = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord, pixel_buffer),
               ks = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
    float weight_spec = (ks.x + ks.y + ks.z) / ((kd.x + kd.y + kd.z) + (ks.x + ks.y + ks.z));
    const float N_dot_O = Dot(rec->wo, rec->normal),
                kr_o = FresnelSchlick(N_dot_O, reflectivity_);
    float kr_i = kr_o,
          pdf_spec = kr_i * weight_spec,
          pdf_diff = (1.0f - kr_i) * (1.0f - weight_spec);
    pdf_spec = pdf_spec / (pdf_spec + pdf_diff);
    pdf_diff = 1.0f - pdf_spec;

    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h(0);
    float D = 0;
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x;
    float N_dot_I = 0.0f;
    if (RandomFloat(seed) < pdf_spec)
    { // 抽样塑料清漆层
        SampleGgx(RandomFloat(seed), RandomFloat(seed), roughness, h, D);
        h = ToWorld(h, rec->normal);
        rec->wi = -Reflect(-rec->wo, h);
        N_dot_I = Dot(-rec->wi, rec->normal);
        if (N_dot_I < kEpsilon)
            return;
        kr_i = FresnelSchlick(N_dot_I, reflectivity_);
        pdf_spec = kr_i * weight_spec, pdf_diff = (1.0f - kr_i) * weight_spec;
        pdf_spec = pdf_spec / (pdf_spec + pdf_diff), pdf_diff = 1.0f - pdf_spec;

        const float H_dot_O = Dot(rec->wo, h);
        pdf_spec *= D / (4.0f * H_dot_O);
        pdf_diff *= PdfHemisCos(ToLocal(-rec->wi, rec->normal));
    }
    else
    { // 抽样塑料基底层
        Vec3 wi_local = Vec3(0);
        float local_pdf_diff = 0.0f;
        SampleHemisCos(RandomFloat(seed), RandomFloat(seed), wi_local, local_pdf_diff);
        rec->wi = -ToWorld(wi_local, rec->normal);
        N_dot_I = Dot(-rec->wi, rec->normal);
        kr_i = FresnelSchlick(N_dot_I, reflectivity_);
        pdf_spec = kr_i * weight_spec, pdf_diff = (1.0f - kr_i) * weight_spec;
        pdf_spec = pdf_spec / (pdf_spec + pdf_diff), pdf_diff = 1.0f - pdf_spec;

        h = Normalize(-rec->wi + rec->wo);
        D = PdfGgx(roughness, rec->normal, h);
        const float H_dot_O = Dot(rec->wo, h);
        pdf_spec *= D / (4.0 * H_dot_O);
        pdf_diff *= local_pdf_diff;
    }
    rec->pdf = pdf_spec + pdf_diff;
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    // 计算塑料清漆层贡献的光能衰减系数
    if (pdf_spec >= kEpsilon)
    {
        const float H_dot_I = Dot(-rec->wi, h),
                    F = FresnelSchlick(H_dot_I, reflectivity_),
                    G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, rec->wo, rec->normal, h));
        Vec3 spec = (F * D * G) / (4.0f * N_dot_O);
        spec *= texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord, pixel_buffer);
        rec->attenuation += spec;
    }

    // 计算塑料基底层贡献的光能衰减系数
    if (pdf_diff >= kEpsilon)
    {
        const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                              pixel_buffer);
        Vec3 diff = albedo * kPiInv * N_dot_I;
        diff *= ((1.0f - kr_i) * (1.0f - kr_o)) / (1.0f - F_avg_);
        rec->attenuation += diff;
    }
}
