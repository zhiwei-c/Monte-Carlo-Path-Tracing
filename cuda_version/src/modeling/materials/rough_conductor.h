#pragma once

#include "material_base.h"

__device__ void Material::InitRoughConductor(bool twosided,
                                             Texture *bump_map,
                                             Texture *opacity_map,
                                             bool mirror,
                                             vec3 eta,
                                             vec3 k,
                                             Texture *specular_reflectance,
                                             MicrofacetDistribType distri,
                                             Texture *alpha_u,
                                             Texture *alpha_v,
                                             float *kulla_conty_table,
                                             float albedo_avg)
{
    type_ = kRoughConductor;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    mirror_ = mirror;
    eta_ = eta;
    k_ = k;
    specular_reflectance_ = specular_reflectance;
    distri_ = distri;
    alpha_u_ = alpha_u;
    alpha_v_ = alpha_v;

    if (albedo_avg < 0)
        return;
    albedo_avg_ = albedo_avg;
    kulla_conty_table_ = kulla_conty_table;

    auto reflectivity = vec3(0),
         edgetint = vec3(0);
    IorToReflectivityEdgetint(eta_, k_, reflectivity, edgetint);

    auto F_avg = AverageFresnelConductor(reflectivity, edgetint);
    f_add_ = F_avg * F_avg * albedo_avg / (vec3(1) - F_avg * (1.0 - albedo_avg));
}

__device__ void Material::SampleRoughConductor(BsdfSampling &bs, const vec3 &sample) const
{
    auto alpha_u = alpha_u_ ? alpha_u_->Color(bs.texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(bs.texcoord).x : 0.1;

    auto h = vec3(0);
    auto D = static_cast<Float>(0);
    SampleNormDistrib(distri_, alpha_u, alpha_v, bs.normal, sample, h, D);

    if (D < kEpsilonPdf)
        return;

    bs.wi = -Reflect(-bs.wo, h);
    auto cos_i_n = myvec::dot(bs.wi, bs.normal);
    if (cos_i_n >= 0)
        return;

    auto jacobian = abs(1.0 / (4.0 * myvec::dot(bs.wo, h)));
    bs.pdf = jacobian * D;
    if (bs.pdf < kEpsilonPdf)
        return;

    auto F = mirror_ ? vec3(1) : FresnelConductor(bs.wi, h, eta_, k_);
    auto G = SmithG1(distri_, alpha_u, alpha_v, -bs.wi, bs.normal, h) *
             SmithG1(distri_, alpha_u, alpha_v, bs.wo, bs.normal, h);
    auto cos_o_n = myvec::dot(bs.wo, bs.normal);
    auto albedo = F * static_cast<Float>(D * G / abs(4.0 * -cos_i_n * cos_o_n));
    if (specular_reflectance_)
        albedo *= specular_reflectance_->Color(bs.texcoord);
    if (albedo_avg_ > 0)
        albedo += EvalMultipleScatter(cos_i_n, cos_o_n);
    bs.attenuation = albedo;

    bs.valid = true;
}

__device__ vec3 Material::EvalRoughConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1;

    auto cos_i_n = abs(myvec::dot(wi, normal)),
         cos_o_n = abs(myvec::dot(wo, normal));

    auto h = myvec::normalize(-wi + wo);

    auto F = mirror_ ? vec3(1) : FresnelConductor(wi, h, eta_, k_);

    auto D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h);

    auto G = SmithG1(distri_, alpha_u, alpha_v, -wi, normal, h) *
             SmithG1(distri_, alpha_u, alpha_v, wo, normal, h);

    auto albedo = F * static_cast<Float>(D * G / (4.0 * cos_i_n * cos_o_n));
    if (specular_reflectance_)
        albedo *= specular_reflectance_->Color(texcoord);

    if (albedo_avg_ > 0)
        albedo += EvalMultipleScatter(cos_i_n, cos_o_n);

    return albedo;
}

__device__ Float Material::PdfRoughConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    // 入射、出射光线需在同侧
    if (NotSameHemis(wo, -wi))
        return 0;

    auto alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;
    auto alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1;

    auto h = myvec::normalize(-wi + wo);

    auto D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h);
    if (D < kEpsilonL)
        return 0;
    else
        return D * abs(1.0 / (4.0 * myvec::dot(wo, h)));
}

__device__ inline void InitRoughConductor(uint m_idx,
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

    material_list[m_idx].InitRoughConductor(material_info_list[m_idx].twosided,
                                            bump_map,
                                            opacity_map,
                                            material_info_list[m_idx].mirror,
                                            material_info_list[m_idx].eta,
                                            material_info_list[m_idx].k,
                                            specular_reflectance,
                                            material_info_list[m_idx].distri,
                                            alpha_u,
                                            alpha_v,
                                            kulla_conty_table,
                                            albedo_avg);
}