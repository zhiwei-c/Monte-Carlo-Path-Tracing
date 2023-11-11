#include "bsdf.cuh"

#include "rtcore.cuh"
#include "utils.cuh"

namespace rt
{

QUALIFIER_D_H void rt::Bsdf::EvaluateDielectric(SamplingRecord *rec) const
{
    float eta = data_.dielectric.eta;
    // 相对折射率的倒数，即入射侧介质和透射侧介质的绝对折射率之比
    float eta_inv = data_.dielectric.eta_inv;
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
    const Texture &roughness_u =
                      data_.texture_buffer[data_.dielectric.id_roughness_u],
                  &roughness_v =
                      data_.texture_buffer[data_.dielectric.id_roughness_v];
    const float alpha_u = roughness_u.GetColor(rec->texcoord).x,
                alpha_v = roughness_v.GetColor(rec->texcoord).x,
                D = PdfGgx(alpha_u, alpha_v, h_local),
                H_dot_I = Dot(-rec->wi, h_world),
                H_dot_O = Dot(rec->wo, h_world),
                F = FresnelSchlick(H_dot_I, data_.dielectric.reflectivity);
    rec->pdf = relfect ? (F * D) / (4.0f * H_dot_O)
                       : (((1.0f - F) * D) *
                          abs(H_dot_O / Sqr(eta_inv * H_dot_I + H_dot_O)));
    if (rec->pdf < kEpsilonFloat)
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

        // const float N_dot_I = Dot(-rec->wi, rec->normal)
        // rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O,
        // roughness);

        const Texture &specular_reflectance =
            data_.texture_buffer[data_.dielectric.id_specular_reflectance];
        const Vec3 spec = specular_reflectance.GetColor(rec->texcoord);
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

        // const float N_dot_I = Dot(rec->normal, -rec->wi);
        // rec->attenuation += EvaluateMultipleScatter(N_dot_I, N_dot_O,
        // roughness, rec->inside, false);

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);

        const Texture &specular_transmittance =
            data_.texture_buffer[data_.dielectric.id_specular_transmittance];
        const Vec3 spec = specular_transmittance.GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
}

QUALIFIER_D_H void rt::Bsdf::SampleDielectric(const Vec3 &xi,
                                              SamplingRecord *rec) const
{
    const float scale = 1.2f - 0.2f * sqrt(abs(Dot(-rec->wo, rec->normal)));
    const Texture &roughness_u =
                      data_.texture_buffer[data_.dielectric.id_roughness_u],
                  &roughness_v =
                      data_.texture_buffer[data_.dielectric.id_roughness_v];
    const float alpha_u = roughness_u.GetColor(rec->texcoord).x * scale,
                alpha_v = roughness_v.GetColor(rec->texcoord).x * scale;

    // 根据GGX法线分布函数重要抽样微平面法线
    Vec3 h_local(0);
    float D = 0;
    SampleGgx(xi.x, xi.y, alpha_u, alpha_v, &h_local, &D);
    const Vec3 h_world = rec->ToWorld(h_local);
    float H_dot_O = Dot(rec->wo, h_world);
    if (H_dot_O < kEpsilonFloat)
        return;

    float eta = data_.dielectric.eta;
    // 相对折射率的倒数，即入射侧介质和透射侧介质的绝对折射率之比
    float eta_inv = data_.dielectric.eta_inv;
    if (!rec->inside)
    { // 如果光线源于物体内部，那么应该颠倒相对折射率
        float temp = eta_inv;
        eta_inv = eta;
        eta = temp;
    }

    Vec3 wt;
    const bool full_reflect = !Ray::Refract(-rec->wo, h_world, eta, &wt);
    float F = FresnelSchlick(H_dot_O, data_.dielectric.reflectivity);
    const Vec3 wo_local = rec->ToLocal(rec->wo);
    if (full_reflect || xi.z < F)
    { // 抽样反射光线
        rec->wi = -Ray::Reflect(-rec->wo, h_world);
        const float N_dot_I = Dot(-rec->wi, rec->normal);
        if (N_dot_I < kEpsilonFloat)
            return;

        rec->pdf = F * D / (4.0f * H_dot_O);
        if (rec->pdf < kEpsilonFloat)
            return;

        const Vec3 wi_local = rec->ToLocal(-rec->wi);
        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                    N_dot_O = wo_local.z;
        rec->attenuation = (F * D * G) / (4.0f * N_dot_O);

        const Texture &specular_reflectance =
            data_.texture_buffer[data_.dielectric.id_specular_reflectance];
        const Vec3 spec = specular_reflectance.GetColor(rec->texcoord);
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
        F = FresnelSchlick(H_dot_I, data_.dielectric.reflectivity);
        rec->pdf =
            ((1.0f - F) * D) * abs(H_dot_O / Sqr(eta_inv * H_dot_I + H_dot_O));
        if (rec->pdf < kEpsilonFloat)
            return;

        const float G = SmithG1Ggx(alpha_u, alpha_v, wi_local, h_local) *
                        SmithG1Ggx(alpha_u, alpha_v, wo_local, h_local),
                    N_dot_O = wo_local.z;
        rec->attenuation =
            (((abs(H_dot_I) * abs(H_dot_O)) * ((1.0f - F) * G * D)) /
             abs(N_dot_O * Sqr(eta_inv * H_dot_I + H_dot_O)));

        // 光线折射后，光路可能覆盖的立体角范围发生了改变，
        //     对辐射亮度进行积分需要进行相应的处理
        rec->attenuation *= Sqr(eta);

        const Texture &specular_transmittance =
            data_.texture_buffer[data_.dielectric.id_specular_transmittance];
        const Vec3 spec = specular_transmittance.GetColor(rec->texcoord);
        rec->attenuation *= spec;
    }
    rec->valid = true;
}

} // namespace rt