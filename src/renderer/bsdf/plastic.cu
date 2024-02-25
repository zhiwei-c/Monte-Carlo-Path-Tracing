#include "csrt/renderer/bsdf.cuh"

#include "csrt/rtcore.cuh"
#include "csrt/utils.cuh"


namespace csrt
{

QUALIFIER_D_H void BSDF::EvaluatePlastic(BSDF::SampleRec *rec) const
{
    // 反射光线与法线方向应该位于同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    if (N_dot_O < kEpsilonFloat)
        return;

    // 计算塑料清漆层和基底层反射的权重
    const Vec3 kd = data_.plastic.diffuse_reflectance->GetColor(rec->texcoord),
               ks = data_.plastic.specular_reflectance->GetColor(rec->texcoord);
    float weight_spec =
        (ks.x + ks.y + ks.z) / ((kd.x + kd.y + kd.z) + (ks.x + ks.y + ks.z));
    const float N_dot_I = Dot(-rec->wi, rec->normal),
                kr_i = FresnelSchlick(N_dot_I, data_.plastic.reflectivity);
    float pdf_spec = kr_i * weight_spec,
          pdf_diff = (1.0f - kr_i) * (1.0f - weight_spec);
    pdf_spec = pdf_spec / (pdf_spec + pdf_diff);
    pdf_diff = 1.0f - pdf_spec;

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const Vec3 h_world = Normalize(-rec->wi + rec->wo),
               h_local = rec->ToLocal(h_world);
    const float alpha = data_.plastic.roughness->GetColor(rec->texcoord).x,
                D = PdfGgx(alpha, h_local), H_dot_O = Dot(rec->wo, h_world);
    pdf_spec *= D / (4.0f * H_dot_O);

    // 反推余弦加权重要抽样时的概率
    const Vec3 wo_local = rec->ToLocal(rec->wo);
    pdf_diff *= wo_local.z;

    // 总概率
    rec->pdf = pdf_spec + pdf_diff;
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    // 计算塑料清漆层贡献的光能衰减系数
    if (pdf_spec > kEpsilon)
    {
        const Vec3 wi_local = rec->ToLocal(-rec->wi);
        const float H_dot_I = Dot(-rec->wi, h_world),
                    F = FresnelSchlick(H_dot_I, data_.plastic.reflectivity),
                    G = (SmithG1Ggx(alpha, wo_local, h_local) *
                         SmithG1Ggx(alpha, wi_local, h_local));
        Vec3 spec = (F * D * G) / (4.0f * N_dot_O);
        rec->attenuation += spec * ks;
    }

    // 计算塑料基底层贡献的光能衰减系数
    if (pdf_diff > kEpsilon)
    {
        Vec3 diff = kd * k1DivPi * N_dot_I;
        const float kr_o = FresnelSchlick(N_dot_O, data_.plastic.reflectivity);
        diff *= ((1.0f - kr_i) * (1.0f - kr_o)) / (1.0f - data_.plastic.F_avg);
        rec->attenuation += diff;
    }
}

QUALIFIER_D_H void BSDF::SamplePlastic(uint32_t *seed,
                                       BSDF::SampleRec *rec) const
{
    // 计算塑料清漆层和基底层反射的权重
    const Vec3 kd = data_.plastic.diffuse_reflectance->GetColor(rec->texcoord),
               ks = data_.plastic.specular_reflectance->GetColor(rec->texcoord);
    float weight_spec =
        (ks.x + ks.y + ks.z) / ((kd.x + kd.y + kd.z) + (ks.x + ks.y + ks.z));
    const float N_dot_O = Dot(rec->wo, rec->normal),
                kr_o = FresnelSchlick(N_dot_O, data_.plastic.reflectivity);
    float kr_i = kr_o, pdf_spec = kr_i * weight_spec,
          pdf_diff = (1.0f - kr_i) * (1.0f - weight_spec);
    pdf_spec = pdf_spec / (pdf_spec + pdf_diff);
    pdf_diff = 1.0f - pdf_spec;

    // 根据GGX法线分布函数重要抽样微平面法线，生成入射光线方向
    Vec3 h_local(0), h_world(0);
    float D = 0;
    const float alpha = data_.plastic.roughness->GetColor(rec->texcoord).x;
    float N_dot_I = 0;
    if (RandomFloat(seed) < pdf_spec)
    { // 抽样塑料清漆层
        SampleGgx(RandomFloat(seed), RandomFloat(seed), alpha, &h_local, &D);
        h_world = rec->ToWorld(h_local);

        rec->wi = -Ray::Reflect(-rec->wo, h_world);
        N_dot_I = Dot(-rec->wi, rec->normal);
        if (N_dot_I < kEpsilonFloat)
            return;

        kr_i = FresnelSchlick(N_dot_I, data_.plastic.reflectivity);
        pdf_spec = kr_i * weight_spec, pdf_diff = (1.0f - kr_i) * weight_spec;
        pdf_spec = pdf_spec / (pdf_spec + pdf_diff), pdf_diff = 1.0f - pdf_spec;

        const float H_dot_O = Dot(rec->wo, h_world);
        pdf_spec *= D / (4.0f * H_dot_O);
        pdf_diff *= Dot(-rec->wi, rec->normal);
    }
    else
    { // 抽样塑料基底层
        Vec3 wi_local = Vec3(0);
        float pdf_diff_local = 0.0f;
        SampleHemisCos(RandomFloat(seed), RandomFloat(seed), &wi_local, &pdf_diff_local);
        rec->wi = -rec->ToWorld(wi_local);

        N_dot_I = Dot(-rec->wi, rec->normal);
        kr_i = FresnelSchlick(N_dot_I, data_.plastic.reflectivity);
        pdf_spec = kr_i * weight_spec, pdf_diff = (1.0f - kr_i) * weight_spec;
        pdf_spec = pdf_spec / (pdf_spec + pdf_diff), pdf_diff = 1.0f - pdf_spec;

        h_world = Normalize(-rec->wi + rec->wo),
        h_local = rec->ToLocal(h_world);
        D = PdfGgx(alpha, h_local);
        const float H_dot_O = Dot(rec->wo, h_world);
        pdf_spec *= D / (4.0 * H_dot_O);
        pdf_diff *= pdf_diff_local;
    }
    rec->pdf = pdf_spec + pdf_diff;
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    // 计算塑料清漆层贡献的光能衰减系数
    if (pdf_spec > kEpsilon)
    {
        const Vec3 wi_local = rec->ToLocal(-rec->wi),
                   wo_local = rec->ToLocal(rec->wo);
        const float H_dot_I = Dot(-rec->wi, h_world),
                    F = FresnelSchlick(H_dot_I, data_.plastic.reflectivity),
                    G = (SmithG1Ggx(alpha, wo_local, h_local) *
                         SmithG1Ggx(alpha, wi_local, h_local));
        Vec3 spec = (F * D * G) / (4.0f * N_dot_O);
        rec->attenuation += spec * ks;
    }

    // 计算塑料基底层贡献的光能衰减系数
    if (pdf_diff > kEpsilon)
    {
        Vec3 diff = kd * k1DivPi * N_dot_I;
        diff *= ((1.0f - kr_i) * (1.0f - kr_o)) / (1.0f - data_.plastic.F_avg);
        rec->attenuation += diff;
    }
}

} // namespace csrt