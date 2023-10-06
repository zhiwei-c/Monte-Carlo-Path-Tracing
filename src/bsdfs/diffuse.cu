#include "diffuse.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void Diffuse::Evaluate(const float *pixel_buffer, Texture **texture_buffer,
                                        uint64_t *seed, SamplingRecord *rec) const
{
    // 反推余弦加权重要抽样时的概率
    rec->pdf = PdfHemisCos(ToLocal(rec->wo, rec->normal));
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                          pixel_buffer);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    rec->attenuation = albedo * kPiInv * N_dot_I;
}

QUALIFIER_DEVICE void Diffuse::Sample(const float *pixel_buffer, Texture **texture_buffer,
                                      uint64_t *seed, SamplingRecord *rec) const
{
    // 余弦加权重要抽样入射光线的方向
    Vec3 wi_local = Vec3(0);
    float pdf = 0.0f;
    SampleHemisCos(RandomFloat(seed), RandomFloat(seed), wi_local, pdf);
    if (pdf < kEpsilon)
        return;

    rec->valid = true;
    rec->wi = -ToWorld(wi_local, rec->normal);
    rec->pdf = pdf;
    const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                          pixel_buffer);
    const float N_dot_I = Dot(-rec->wi, rec->normal);
    rec->attenuation = albedo * kPiInv * N_dot_I;
}