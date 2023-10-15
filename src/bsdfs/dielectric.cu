#include "dielectric.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE void Dielectric::Evaluate(const float *pixel_buffer,
                                                Texture **texture_buffer,
                                                uint64_t *seed, SamplingRecord *rec) const
{
    float eta = eta_,
          eta_inv = eta_inv_; // 相对折射率的倒数，即入射侧介质和透射侧介质的绝对折射率之比
    if (rec->inside)
    { // 如果光线源于物体内部，那么应该颠倒相对折射率
        float temp = eta_inv;
        eta_inv = eta;
        eta = temp;
    }

    // 计算微平面法线，使之与入射光线同侧
    const float N_dot_O = Dot(rec->wo, rec->normal);
    const bool relfect = N_dot_O > 0.0f;
    Vec3 h = relfect ? Normalize(-rec->wi + rec->wo) : -Normalize(eta_inv * (-rec->wi) + rec->wo);

    // 反推根据GGX法线分布函数重要抽样微平面法线的概率
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x,
                H_dot_I = Dot(-rec->wi, h),
                H_dot_O = Dot(rec->wo, h),
                D = PdfGgx(roughness, rec->normal, h),
                F = FresnelSchlick(H_dot_I, reflectivity_);
    if (relfect)
    {
        rec->pdf = (F * D) / (4.0f * H_dot_O);
    }
    else
    {
        rec->pdf = (((1.0f - F) * D) * abs(H_dot_O / pow(eta_inv * H_dot_I + H_dot_O, 2)));
    }
    if (rec->pdf < kEpsilon)
        return;

    rec->valid = true;
    if (relfect)
    {
        const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, rec->wo, rec->normal, h));
        rec->attenuation = (F * D * G) / abs(4.0 * N_dot_O);

        const float N_dot_I = Dot(rec->normal, -rec->wi);
        rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness, rec->inside, true);

        const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord,
                                                                             pixel_buffer);
        rec->attenuation *= spec;
    }
    else
    {
        const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, -rec->wo, rec->normal, h));
        rec->attenuation = ((abs(H_dot_I) * abs(H_dot_O)) * ((1.0f - F) * G * D)) /
                           abs(N_dot_O * pow(eta_inv * H_dot_I + H_dot_O, 2));

        const float N_dot_I = Dot(rec->normal, -rec->wi);
        rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness, rec->inside, false);

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= eta * eta;
        const Vec3 spec = texture_buffer[id_specular_transmittance_]->GetColor(rec->texcoord,
                                                                               pixel_buffer);
        rec->attenuation *= spec;
    }
}

QUALIFIER_DEVICE void Dielectric::Sample(const float *pixel_buffer,
                                              Texture **texture_buffer,
                                              uint64_t *seed, SamplingRecord *rec) const
{
    const float scale = 1.2f - 0.2f * sqrt(abs(Dot(-rec->wo, rec->normal)));
    const float roughness = texture_buffer[id_roughness_]->GetColor(rec->texcoord, pixel_buffer).x;
    const float roughness_scaled = roughness * scale;

    // 根据GGX法线分布函数重要抽样微平面法线
    Vec3 h(0);
    float D = 0;
    SampleGgx(RandomFloat(seed), RandomFloat(seed), roughness_scaled, h, D);
    h = ToWorld(h, rec->normal);

    float H_dot_O = Dot(rec->wo, h);
    if (H_dot_O < kEpsilon)
        return;

    float eta_inv = eta_inv_,
            eta = eta_; // 相对折射率，即透射侧介质与入射侧介质的绝对折射率之比
    if (!rec->inside)
    { // 如果发生折射，那么光线源于物体内部
        // 需要颠倒相对折射率，使宏观法线和微平面法线与入射光线同侧
        float temp = eta_inv;
        eta_inv = eta;
        eta = temp;
    }

    bool full_reflect = !Refract(-rec->wo, h, eta, &rec->wi);
    float F = FresnelSchlick(H_dot_O, reflectivity_);
    if (full_reflect || RandomFloat(seed) < F)
    { // 抽样反射光线
        rec->wi = -Reflect(-rec->wo, h);
        const float N_dot_I = Dot(-rec->wi, rec->normal);
        if (N_dot_I < kEpsilon)
            return;

        rec->pdf = F * D / (4.0f * H_dot_O);
        if (rec->pdf < kEpsilon)
            return;

        const float G = (SmithG1Ggx(roughness, -rec->wi, rec->normal, h) *
                         SmithG1Ggx(roughness, rec->wo, rec->normal, h)),
                    N_dot_O = Dot(rec->wo, rec->normal);
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

        rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness, rec->inside, true);

        const Vec3 spec = texture_buffer[id_specular_reflectance_]->GetColor(rec->texcoord,
                                                                             pixel_buffer);
        rec->attenuation *= spec;
    }
    else
    { // 抽样折射光线
        rec->wi = -rec->wi;

        const float N_dot_I = Dot(-rec->wi, -rec->normal);
        if (N_dot_I <= kEpsilon)
            return;

        const float H_dot_I = Dot(-rec->wi, -h);
        if (H_dot_I <= kEpsilon)
            return;

        H_dot_O = -H_dot_O;
        F = FresnelSchlick(H_dot_I, reflectivity_);
        rec->pdf = ((1.0f - F) * D) * abs(H_dot_O / pow(eta_inv * H_dot_I + H_dot_O, 2));
        if (rec->pdf < kEpsilon)
            return;

        float G = (SmithG1Ggx(roughness, -rec->wi, -rec->normal, -h) *
                   SmithG1Ggx(roughness, rec->wo, rec->normal, h)),
              N_dot_O = Dot(rec->wo, rec->normal);
        rec->attenuation = (abs(H_dot_I * H_dot_O) * ((1.0f - F) * G * D)) /
                           abs(N_dot_O * pow(eta_inv * H_dot_I + H_dot_O, 2));

        rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O, roughness, !rec->inside, false);

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= eta * eta;

        const Vec3 spec = texture_buffer[id_specular_transmittance_]->GetColor(rec->texcoord,
                                                                               pixel_buffer);
        rec->attenuation *= spec;
    }
    rec->valid = true;
}

QUALIFIER_DEVICE float Dielectric::EvaluateMultipleScatter(const float N_dot_I,
                                                                const float N_dot_O,
                                                                const float roughness,
                                                                const bool inside,
                                                                const bool reflect) const
{
    const float brdf_i = GetBrdfAvg(N_dot_I, roughness),
                brdf_o = GetBrdfAvg(N_dot_O, roughness),
                albedo_avg = GetAlbedoAvg(roughness),
                f_ms = (1.0f - brdf_i) * (1.0f - brdf_o) / (kPi * (1.0f - albedo_avg));

    const float F_avg = inside ? F_avg_inv_ : F_avg_,
                eta = inside ? eta_inv_ : eta_;

    const float f_add = pow(F_avg, 2) * albedo_avg / (1.0f - F_avg * (1.0f - albedo_avg)),
                ratio_trans = ((1.0f - F_avg_) * (1.0f - F_avg_inv_) * pow(eta, 2) /
                               ((1.0f - F_avg_) + (1.0f - F_avg_inv_) * pow(eta, 2)));
    const float ret = f_ms * f_add * N_dot_I;
    return reflect ? (1.0f - ratio_trans) * ret : ratio_trans * ret;
}