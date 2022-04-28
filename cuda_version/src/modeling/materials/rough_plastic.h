#pragma once

#include "material_base.h"

__global__ void InitRoughPlastic(uint m_idx,
                                 float *kulla_conty_table,
                                 float albedo_avg,
                                 MaterialInfo *material_info_list,
                                 Texture *texture_list,
                                 Material *material_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        auto bump_map = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].bump_map_idx != kUintMax)
            bump_map = texture_list + material_info_list[m_idx].bump_map_idx;

        auto opacity_map = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].opacity_idx != kUintMax)
            opacity_map = texture_list + material_info_list[m_idx].opacity_idx;

        auto diffuse_reflectance = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].diffuse_reflectance_idx != kUintMax)
            diffuse_reflectance = texture_list + material_info_list[m_idx].diffuse_reflectance_idx;

        auto specular_reflectance = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].specular_reflectance_idx != kUintMax)
            specular_reflectance = texture_list + material_info_list[m_idx].specular_reflectance_idx;

        auto alpha = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].alpha_u_idx != kUintMax)
            alpha = texture_list + material_info_list[m_idx].alpha_u_idx;

        material_list[m_idx].InitRoughPlastic(material_info_list[m_idx].twosided,
                                              bump_map,
                                              opacity_map,
                                              material_info_list[m_idx].eta,
                                              diffuse_reflectance,
                                              specular_reflectance,
                                              material_info_list[m_idx].distri,
                                              alpha,
                                              material_info_list[m_idx].nonlinear,
                                              kulla_conty_table,
                                              albedo_avg);
    }
}

__device__ void Material::InitRoughPlastic(bool twosided,
                                           Texture *bump_map,
                                           Texture *opacity_map,
                                           vec3 eta,
                                           Texture *diffuse_reflectance,
                                           Texture *specular_reflectance,
                                           MicrofacetDistribType distri,
                                           Texture *alpha,
                                           bool nonlinear,
                                           float *kulla_conty_table,
                                           float albedo_avg)
{
    type_ = kRoughPlastic;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    eta_d_ = eta.x;
    eta_inv_d_ = 1.0 / eta.x;
    diffuse_reflectance_ = diffuse_reflectance;
    specular_reflectance_ = specular_reflectance;
    distri_ = distri;
    alpha_u_ = alpha;
    alpha_v_ = alpha;
    nonlinear_ = nonlinear;
    fdr_int_ = AverageFresnel(1.0 / eta.x);
    fdr_ext_ = AverageFresnel(eta.x);

    if (albedo_avg < 0)
        return;
    albedo_avg_ = albedo_avg;
    kulla_conty_table_ = kulla_conty_table;

    f_add_ = vec3(fdr_ext_ * fdr_ext_ * albedo_avg_ / (1.0 - fdr_ext_ * (1.0 - albedo_avg_)));
}

__device__ void Material::SampleRoughPlastic(BsdfSampling &bs, const vec3 &sample) const
{
    auto alpha = alpha_u_ ? alpha_u_->Color(bs.texcoord).x : 0.1;

    auto specular_sampling_weight = SpecularSamplingWeight(bs.texcoord);

    auto kr = Fresnel(-bs.wo, bs.normal, eta_inv_d_);
    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

    if (sample.z < pdf_specular)
    {
        auto facet_normal = vec3(0);
        Float pdf = 0;
        SampleNormDistrib(distri_, alpha, alpha, bs.normal, sample, facet_normal, pdf);
        bs.wi = -Reflect(-bs.wo, facet_normal);
        if (myvec::dot(bs.wi, bs.normal) >= 0)
            return;
    }
    else
    {
        auto wi_local = vec3(0);
        Float pdf = 0;
        HemisCos(sample.x, sample.y, wi_local, pdf);
        bs.wi = -ToWorld(wi_local, bs.normal);
    }

    bs.pdf = PdfRoughPlastic(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    if (bs.pdf < kEpsilonL)
        return;

    bs.attenuation = EvalRoughPlastic(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    bs.valid = true;
}

__device__ vec3 Material::EvalRoughPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto alpha = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;

    auto cos_i_n = myvec::dot(wi, normal);
    auto cos_o_n = myvec::dot(wo, normal);

    auto albedo = vec3(0);

    auto diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(texcoord) : vec3(0.5);
    if (nonlinear_)
        albedo = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
    else
        albedo = diffuse_reflectance / (1.0 - fdr_int_);

    auto kr_i = Fresnel(wi, normal, eta_inv_d_);
    auto kr_o = Fresnel(-wo, normal, eta_inv_d_);
    albedo *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

    auto h = myvec::normalize(-wi + wo);
    auto F = Fresnel(wi, h, eta_inv_d_);
    auto D = PdfNormDistrib(distri_, alpha, alpha, normal, h);

    if (D > kEpsilon)
    {
        auto G = SmithG1(distri_, alpha, alpha, -wi, normal, h) *
                 SmithG1(distri_, alpha, alpha, wo, normal, h);
        auto attenuation = vec3(F * D * G / (4.0 * abs(cos_i_n * cos_o_n)));

        if (albedo_avg_ > 0)
            attenuation += EvalMultipleScatter(cos_i_n, cos_o_n);

        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        albedo += attenuation;
    }

    return albedo;
}
__device__ Float Material::PdfRoughPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    if (NotSameHemis(wo, normal))
        return 0;

    if (myvec::dot(wi, normal) * myvec::dot(wo, normal) >= 0)
        return 0;

    auto alpha = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1;

    auto kr = Fresnel(wi, normal, eta_inv_d_);
    auto specular_sampling_weight = SpecularSamplingWeight(texcoord);

    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    pdf_diffuse = 1.0 - pdf_specular;

    auto wo_local = ToLocal(wo, normal);
    auto pdf = pdf_diffuse * PdfHemisCos(wo_local);

    auto h = myvec::normalize(-wi + wo);

    auto D = PdfNormDistrib(distri_, alpha, alpha, normal, h);

    if (D > kEpsilon)
    {
        auto jacobian = abs(1.0 / (4.0 * myvec::dot(wo, h)));
        pdf += pdf_specular * D * jacobian;
    }
    return pdf;
}
