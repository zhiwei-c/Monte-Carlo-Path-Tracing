#include "rough_diffuse.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void RoughDiffuse::Evaluate(Texture **texture_buffer, const float *pixel_buffer, 
                                             uint32_t *seed, SamplingRecord *rec) const
{
    // 反推余弦加权重要抽样时的概率
    rec->pdf = PdfHemisCos(ToLocal(rec->wo, rec->normal));
    if (rec->pdf < kEpsilon)
        return;
    else
        rec->valid = true;

    /* Conversion from Beckmann-style RMS roughness to
    Oren-Nayar-style slope-area variance. The factor
    of 1/sqrt(2) was found to be a perfect fit up
    to extreme roughness values (>.5), after which
    the match is not as good anymore */
    const float conversion_factor = 1.0f / sqrtf(2.0f);

    const float sigma = (texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x *
                         conversion_factor),
                sigma_2 = sigma * sigma;

    const float N_dot_I = Dot(-rec->wi, rec->normal),
                N_dot_O = Dot(rec->wo, rec->normal),
                sin_theta_i = sqrtf(1.0f - N_dot_I * N_dot_I),
                sin_theta_o = sqrtf(1.0f - N_dot_O * N_dot_O);

    float phi_i, theta_i;
    CartesianToSpherical(ToLocal(-rec->wi, rec->normal), &theta_i, &phi_i, nullptr);

    float phi_o, theta_o;
    CartesianToSpherical(ToLocal(rec->wo, rec->normal), &theta_o, &phi_o, nullptr);

    float cos_phi_diff = cosf(phi_i) * cosf(phi_o) + sinf(phi_i) * sinf(phi_o);

    const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                          pixel_buffer);
    if (use_fast_approx_)
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
        rec->attenuation = albedo * kPiInv * N_dot_I *
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
              tmp2 = 4.0f * kPiInv * kPiInv * alpha * beta,
              tmp3 = 2.0f * beta * kPiInv;

        float C1 = 1.0f - 0.5f * sigma_2 / (sigma_2 + 0.33f),
              C2 = 0.45f * tmp,
              C3 = 0.125f * tmp * tmp2 * tmp2,
              C4 = 0.17f * sigma_2 / (sigma_2 + 0.13f);

        if (cos_phi_diff > 0)
            C2 *= sin_alpha;
        else
            C2 *= sin_alpha - tmp3 * tmp3 * tmp3;

        float tan_half = (sin_alpha + sin_beta) /
                         (sqrtf(fmaxf(0.0f, 1.0f - sin_alpha * sin_alpha)) +
                          sqrt(fmaxf(0.0f, 1.0f - sin_beta * sin_beta)));

        Vec3 sngl_scat = albedo * (C1 + cos_phi_diff * C2 * tan_beta +
                                   (1.0f - fabs(cos_phi_diff)) * C3 * tan_half),
             dbl_scat = albedo * albedo * (C4 * (1.0f - cos_phi_diff * tmp3 * tmp3));
        rec->attenuation = (sngl_scat + dbl_scat) * kPiInv * N_dot_I;
    }
}

QUALIFIER_DEVICE void RoughDiffuse::Sample(Texture **texture_buffer, const float *pixel_buffer, 
                                           uint32_t *seed, SamplingRecord *rec) const
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

    /* Conversion from Beckmann-style RMS roughness to
    Oren-Nayar-style slope-area variance. The factor
    of 1/sqrt(2) was found to be a perfect fit up
    to extreme roughness values (>.5), after which
    the match is not as good anymore */
    const float conversion_factor = 1.0f / sqrtf(2.0f);
    
    const float sigma = (texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x *
                         conversion_factor),
                sigma_2 = sigma * sigma;

    const float N_dot_I = Dot(-rec->wi, rec->normal),
                N_dot_O = Dot(rec->wo, rec->normal),
                sin_theta_i = sqrtf(1.0f - N_dot_I * N_dot_I),
                sin_theta_o = sqrtf(1.0f - N_dot_O * N_dot_O);

    float phi_i, theta_i;
    CartesianToSpherical(ToLocal(-rec->wi, rec->normal), &theta_i, &phi_i, nullptr);

    float phi_o, theta_o;
    CartesianToSpherical(ToLocal(rec->wo, rec->normal), &theta_o, &phi_o, nullptr);

    float cos_phi_diff = cosf(phi_i) * cosf(phi_o) + sinf(phi_i) * sinf(phi_o);

    const Vec3 albedo = texture_buffer[id_diffuse_reflectance_]->GetColor(rec->texcoord,
                                                                          pixel_buffer);
    if (use_fast_approx_)
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
        rec->attenuation = albedo * kPiInv * N_dot_I *
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
              tmp2 = 4.0f * kPiInv * kPiInv * alpha * beta,
              tmp3 = 2.0f * beta * kPiInv;

        float C1 = 1.0f - 0.5f * sigma_2 / (sigma_2 + 0.33f),
              C2 = 0.45f * tmp,
              C3 = 0.125f * tmp * tmp2 * tmp2,
              C4 = 0.17f * sigma_2 / (sigma_2 + 0.13f);

        if (cos_phi_diff > 0)
            C2 *= sin_alpha;
        else
            C2 *= sin_alpha - tmp3 * tmp3 * tmp3;

        float tan_half = (sin_alpha + sin_beta) /
                         (sqrtf(fmaxf(0.0f, 1.0f - sin_alpha * sin_alpha)) +
                          sqrt(fmaxf(0.0f, 1.0f - sin_beta * sin_beta)));

        Vec3 sngl_scat = albedo * (C1 + cos_phi_diff * C2 * tan_beta +
                                   (1.0f - fabs(cos_phi_diff)) * C3 * tan_half),
             dbl_scat = albedo * albedo * (C4 * (1.0f - cos_phi_diff * tmp3 * tmp3));
        rec->attenuation = (sngl_scat + dbl_scat) * kPiInv * N_dot_I;
    }
}