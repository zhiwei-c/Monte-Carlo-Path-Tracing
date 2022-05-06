#pragma once

#include "material_base.h"

__device__ void Material::InitRoughDielectric(bool twosided,
                                              Texture *bump_map,
                                              Texture *opacity_map,
                                              vec3 eta,
                                              Texture *specular_reflectance,
                                              Texture *specular_transmittance,
                                              MicrofacetDistribType distri,
                                              Texture *alpha_u,
                                              Texture *alpha_v,
                                              float *kulla_conty_table,
                                              float albedo_avg)
{
    type_ = kRoughDielectric;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    eta_d_ = eta.x;
    eta_inv_d_ = 1.0 / eta.x;
    specular_reflectance_ = specular_reflectance;
    specular_transmittance_ = specular_transmittance;
    distri_ = distri;
    alpha_u_ = alpha_u;
    alpha_v_ = alpha_v;

    if (albedo_avg < 0)
        return;
    albedo_avg_ = albedo_avg;
    kulla_conty_table_ = kulla_conty_table;

    auto F_avg = AverageFresnel(eta_d_);
    f_add_ = vec3(F_avg * albedo_avg / (1.0 - F_avg * (1.0 - albedo_avg)));

    auto F_avg_inv = AverageFresnel(eta_inv_d_);
    f_add_inv_ = vec3(F_avg_inv * albedo_avg / (1.0 - F_avg_inv * (1.0 - albedo_avg)));

    ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * eta_d_ * eta_d_ /
               ((1.0 - F_avg) + (1.0 - F_avg_inv) * eta_d_ * eta_d_);

    ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * eta_inv_d_ * eta_inv_d_ /
                   ((1.0 - F_avg_inv) + (1.0 - F_avg) * eta_inv_d_ * eta_inv_d_);
}

__device__ void Material::SampleRoughDielectric(BsdfSampling &bs, const vec3 &sample) const
{
    auto eta = bs.inside == kTrue ? eta_inv_d_ : eta_d_;     //相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
    auto eta_inv = bs.inside == kTrue ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
    auto ratio_t = bs.inside == kTrue ? ratio_t_inv_ : ratio_t_;
    auto ratio_t_inv = bs.inside == kTrue ? ratio_t_ : ratio_t_inv_;

    auto alpha_u = alpha_u_ ? alpha_u_->Color(bs.texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(bs.texcoord).x : 0.1;

    // Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
    auto scale = 1.2 - 0.2 * sqrt(abs(myvec::dot(-bs.wo, bs.normal)));
    alpha_u *= scale;
    alpha_v *= scale;

    auto h = vec3(0);
    auto D = static_cast<Float>(0);
    SampleNormDistrib(distri_, alpha_u, alpha_v, bs.normal, sample, h, D);

    if (D < kEpsilonPdf)
        return;

    auto F = Fresnel(-bs.wo, h, eta_inv);
    if (sample.z < F)
    {
        bs.wi = -Reflect(-bs.wo, h);
        auto cos_i_n = myvec::dot(bs.wi, bs.normal);
        if (cos_i_n >= 0)
            return;

        auto jacobian = abs(1.0 / (4.0 * myvec::dot(bs.wo, h)));
        bs.pdf = F * D * jacobian;
        if (bs.pdf < kEpsilonPdf || albedo_avg_ > 0 && alpha_u > 0.01 && alpha_v > 0.01 && bs.pdf < kEpsilonPdfL)
            return;

        auto G = SmithG1(distri_, alpha_u, alpha_v, -bs.wi, bs.normal, h) *
                 SmithG1(distri_, alpha_u, alpha_v, bs.wo, bs.normal, h);
        auto cos_o_n = myvec::dot(bs.wo, bs.normal);
        bs.attenuation = vec3(F * D * G / (4.0 * abs(cos_i_n * cos_o_n)));
        if (specular_reflectance_)
            bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
        if (albedo_avg_ > 0)
            bs.attenuation += (1 - ratio_t) * EvalMultipleScatter(cos_i_n, cos_o_n, bs.inside);
    }
    else
    {
        bs.wi = -Refract(-bs.wo, h, eta_inv);
        auto cos_i_n = myvec::dot(bs.wi, bs.normal);
        if (cos_i_n <= 0)
            return;

        bs.normal = -bs.normal;
        bs.inside = !bs.inside;
        h = -h;
        eta_inv = eta;
        ratio_t = ratio_t_inv;

        F = Fresnel(bs.wi, h, eta_inv);
        auto cos_i_h = myvec::dot(-bs.wi, h),
             cos_o_h = myvec::dot(bs.wo, h);
        auto jacobian = abs(cos_o_h / pow(eta_inv * cos_i_h + cos_o_h, 2));
        bs.pdf = (1.0 - F) * D * jacobian;
        if (bs.pdf < kEpsilonPdf || albedo_avg_ > 0 && alpha_u > 0.01 && alpha_v > 0.01 && bs.pdf < kEpsilonPdfL)
            return;

        auto G = SmithG1(distri_, alpha_u, alpha_v, -bs.wi, bs.normal, h) *
                 SmithG1(distri_, alpha_u, alpha_v, bs.wo, bs.normal, h);
        auto cos_o_n = myvec::dot(bs.wo, bs.normal);
        bs.attenuation = vec3(abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                  (cos_i_n * cos_o_n * pow(eta_inv * cos_i_h + cos_o_h, 2))));
        if (specular_transmittance_)
            bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
        if (albedo_avg_ > 0)
            bs.attenuation += ratio_t * EvalMultipleScatter(cos_i_n, cos_o_n, bs.inside);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        bs.attenuation *= eta_inv * eta_inv;
    }
    bs.valid = true;
}

__device__ vec3 Material::EvalRoughDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto eta_inv = inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
    auto ratio_t = inside ? ratio_t_inv_ : ratio_t_;

    auto alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1;

    auto cos_o_n = myvec::dot(wo, normal);
    auto cos_i_n = myvec::dot(-wi, normal);

    auto h = vec3(0);
    auto F = static_cast<Float>(0);
    auto relfect = cos_o_n > 0;
    if (relfect)
    {
        h = myvec::normalize(-wi + wo);
        F = Fresnel(wi, h, eta_inv);
    }
    else
    {
        h = myvec::normalize(-eta_inv * wi + wo);
        if (NotSameHemis(h, normal))
            h = -h;
        F = Fresnel(wi, h, eta_inv);
    }

    auto D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h);

    auto G = SmithG1(distri_, alpha_u, alpha_v, -wi, normal, h) *
             SmithG1(distri_, alpha_u, alpha_v, wo, normal, h);

    if (relfect)
    {
        auto attenuation = vec3(F * D * G / (4.0 * abs(cos_i_n * cos_o_n)));
        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        if (albedo_avg_ > 0)
            attenuation += (1 - ratio_t) * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
        return attenuation;
    }
    else
    {
        auto cos_i_h = myvec::dot(-wi, h),
             cos_o_h = myvec::dot(wo, h);
        auto attenuation = vec3(abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                    (cos_i_n * cos_o_n * pow(eta_inv * cos_i_h + cos_o_h, 2))));
        if (specular_transmittance_)
            attenuation *= specular_transmittance_->Color(texcoord);
        if (albedo_avg_ > 0)
            attenuation += ratio_t * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        attenuation *= eta_inv * eta_inv;
        return attenuation;
    }
}
__device__ Float Material::PdfRoughDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto eta_inv = inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1;

    auto cos_i_n = myvec::dot(-wi, normal),
         cos_o_n = myvec::dot(wo, normal);

    auto h = vec3(0);
    auto relfect = cos_o_n > 0;
    if (relfect)
    {
        h = myvec::normalize(-wi + wo);
    }
    else
    {
        h = myvec::normalize(-eta_inv * wi + wo);
        if (NotSameHemis(h, normal))
            h = -h;
    }

    auto F = Fresnel(wi, h, eta_inv);

    auto D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h);
    if (D < kEpsilon)
        return 0;

    if (relfect)
    {
        auto jacobian = abs(1.0 / (4.0 * myvec::dot(wo, h)));
        return F * D * jacobian;
    }
    else
    {
        auto jacobian = abs(myvec::dot(wo, h) /
                            pow(eta_inv * myvec::dot(-wi, h) + myvec::dot(wo, h), 2));
        return (1.0 - F) * D * jacobian;
    }
}

__device__ inline void InitRoughDielectric(uint m_idx,
                                           MaterialInfo *material_info_list,
                                           Texture *texture_list,
                                           Material *material_list)
{
    auto bump_map = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + material_info_list[m_idx].bump_map_idx;

    auto opacity_map = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + material_info_list[m_idx].opacity_idx;

    auto specular_reflectance = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].specular_reflectance_idx != kUintMax)
        specular_reflectance = texture_list + material_info_list[m_idx].specular_reflectance_idx;

    auto specular_transmittance = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].specular_transmittance_idx != kUintMax)
        specular_transmittance = texture_list + material_info_list[m_idx].specular_transmittance_idx;

    auto alpha_u = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].alpha_u_idx != kUintMax)
        alpha_u = texture_list + material_info_list[m_idx].alpha_u_idx;

    auto alpha_v = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].alpha_v_idx != kUintMax)
        alpha_v = texture_list + material_info_list[m_idx].alpha_v_idx;

    auto albedo_avg = static_cast<float>(-1);
    auto kulla_conty_table = static_cast<float *>(nullptr);
    CreateCosinAlbedoTexture(material_info_list[m_idx].distri, alpha_u, alpha_v,
                             kulla_conty_table, albedo_avg);

    material_list[m_idx].InitRoughDielectric(material_info_list[m_idx].twosided,
                                             bump_map,
                                             opacity_map,
                                             material_info_list[m_idx].eta,
                                             specular_reflectance,
                                             specular_transmittance,
                                             material_info_list[m_idx].distri,
                                             alpha_u,
                                             alpha_v,
                                             kulla_conty_table,
                                             albedo_avg);
}
