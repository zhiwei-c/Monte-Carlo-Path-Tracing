#include "csrt/renderer/bsdfs/rough_diffuse.cuh"

#include "csrt/renderer/bsdfs/bsdf.cuh"
#include "csrt/utils.cuh"

namespace
{
using namespace csrt;

QUALIFIER_D_H void Evaluate(const float rougness, const Vec3 &albedo,
                            const bool use_fast_approx, BsdfSampleRec *rec)
{
    /* Conversion from Beckmann-style RMS roughness to
    Oren-Nayar-style slope-area variance. The factor
    of 1/sqrt(2) was found to be a perfect fit up
    to extreme roughness values (>.5), after which
    the match is not as good anymore */
    constexpr float conversion_factor = 0.70710678118f;
    const float sigma_2 = Sqr(rougness * conversion_factor);

    const Vec3 wi_local = rec->ToLocal(-rec->wi),
               wo_local = rec->ToLocal(rec->wo);
    const float N_dot_I = wi_local.z, N_dot_O = wo_local.z,
                sin_theta_i = sqrtf(1.0f - N_dot_I * N_dot_I),
                sin_theta_o = sqrtf(1.0f - N_dot_O * N_dot_O);

    float phi_i, theta_i;
    CartesianToSpherical(wi_local, &theta_i, &phi_i, nullptr);
    float phi_o, theta_o;
    CartesianToSpherical(wo_local, &theta_o, &phi_o, nullptr);
    float cos_phi_diff = cosf(phi_i) * cosf(phi_o) + sinf(phi_i) * sinf(phi_o);

    if (use_fast_approx)
    {

        float A = 1.0f - 0.5f * sigma_2 / (sigma_2 + 0.33f),
              B = 0.45f * sigma_2 / (sigma_2 + 0.09f);
        float sin_alpha, tan_beta;
        if (N_dot_I > N_dot_O)
        {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / N_dot_I;
        }
        else
        {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / N_dot_O;
        }
        rec->attenuation =
            albedo * k1DivPi * N_dot_I *
            (A + B * fmaxf(cos_phi_diff, 0.0f) * sin_alpha * tan_beta);
    }
    else
    {
        float alpha = fmaxf(theta_i, theta_o), beta = fminf(theta_i, theta_o);
        float sin_alpha, sin_beta, tan_beta;
        if (N_dot_I > N_dot_O)
        {
            sin_alpha = sin_theta_o;
            sin_beta = sin_theta_i;
            tan_beta = sin_theta_i / N_dot_I;
        }
        else
        {
            sin_alpha = sin_theta_i;
            sin_beta = sin_theta_o;
            tan_beta = sin_theta_o / N_dot_O;
        }

        float tmp = sigma_2 / (sigma_2 + 0.09f),
              tmp2 = 4.0f * k1DivPi * k1DivPi * alpha * beta,
              tmp3 = 2.0f * beta * k1DivPi;

        float C1 = 1.0f - 0.5f * sigma_2 / (sigma_2 + 0.33f), C2 = 0.45f * tmp,
              C3 = 0.125f * tmp * tmp2 * tmp2,
              C4 = 0.17f * sigma_2 / (sigma_2 + 0.13f);

        if (cos_phi_diff > 0)
            C2 *= sin_alpha;
        else
            C2 *= sin_alpha - pow(tmp3, 3);

        float tan_half = (sin_alpha + sin_beta) /
                         (sqrtf(fmaxf(0.0f, 1.0f - Sqr(sin_alpha))) +
                          sqrt(fmaxf(0.0f, 1.0f - Sqr(sin_beta))));

        Vec3 sngl_scat = albedo * (C1 + cos_phi_diff * C2 * tan_beta +
                                   (1.0f - fabs(cos_phi_diff)) * C3 * tan_half),
             dbl_scat = Sqr(albedo) * (C4 * (1.0f - cos_phi_diff * Sqr(tmp3)));
        rec->attenuation = (sngl_scat + dbl_scat) * k1DivPi * N_dot_I;
    }
}

} // namespace

namespace csrt
{

QUALIFIER_D_H void EvaluateRoughDiffuse(const RoughDiffuseData &data,
                                        BsdfSampleRec *rec)
{
    // 反推余弦加权重要抽样时的概率
    rec->pdf = Dot(rec->wo, rec->normal);
    if (rec->pdf < kEpsilon)
        return;
    rec->valid = true;

    const float alpha = data.roughness->GetColor(rec->texcoord).x;
    const Vec3 albedo = data.diffuse_reflectance->GetColor(rec->texcoord);
    ::Evaluate(alpha, albedo, data.use_fast_approx, rec);
}

QUALIFIER_D_H void SampleRoughDiffuse(const RoughDiffuseData &data,
                                      uint32_t *seed, BsdfSampleRec *rec)
{
    // 余弦加权重要抽样入射光线的方向
    Vec3 wi;
    SampleHemisCos(RandomFloat(seed), RandomFloat(seed), &wi, &rec->pdf);
    if (rec->pdf < kEpsilon)
        return;

    rec->wi = -Normalize(wi.x * rec->tangent + wi.y * rec->bitangent +
                         wi.z * rec->normal);
    rec->valid = true;

    const float alpha = data.roughness->GetColor(rec->texcoord).x;
    const Vec3 albedo = data.diffuse_reflectance->GetColor(rec->texcoord);
    ::Evaluate(alpha, albedo, data.use_fast_approx, rec);
}

} // namespace csrt