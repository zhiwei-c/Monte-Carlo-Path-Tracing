#pragma once

#include "material_base.h"

__global__ void InitPlastic(uint m_idx,
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

        material_list[m_idx].InitPlastic(material_info_list[m_idx].twosided,
                                         bump_map,
                                         opacity_map,
                                         material_info_list[m_idx].eta,
                                         diffuse_reflectance,
                                         specular_reflectance,
                                         material_info_list[m_idx].nonlinear);
    }
}
__device__ void Material::InitPlastic(bool twosided,
                                      Texture *bump_map,
                                      Texture *opacity_map,
                                      vec3 eta,
                                      Texture *diffuse_reflectance,
                                      Texture *specular_reflectance,
                                      bool nonlinear)
{
    type_ = kPlastic;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    eta_d_ = eta.x;
    eta_inv_d_ = 1.0 / eta.x;
    diffuse_reflectance_ = diffuse_reflectance;
    specular_reflectance_ = specular_reflectance;
    nonlinear_ = nonlinear;
    fdr_int_ = FresnelDiffuseReflectance(1.0 / eta.x);
    fdr_ext_ = FresnelDiffuseReflectance(eta.x);
}

__device__ void Material::SamplePlastic(BsdfSampling &bs, const vec3 &sample) const
{
    auto eta_inv = bs.inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto specular_sampling_weight = SpecularSamplingWeight(bs.texcoord);

    auto kr = Fresnel(-bs.wo, bs.normal, eta_inv);
    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

    if (sample.x < pdf_specular)
        bs.wi = -Reflect(-bs.wo, bs.normal);
    else
    {
        auto wi_local = vec3(0);
        auto pdf = static_cast<Float>(0);
        HemisCos(sample.y, sample.z, wi_local, pdf);
        bs.wi = -ToWorld(wi_local, bs.normal);
    }

    bs.pdf = PdfPlastic(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    if (bs.pdf < kEpsilonPdf)
        return;

    bs.attenuation = EvalPlastic(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    bs.valid = true;
}

__device__ vec3 Material::EvalPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    if (NotSameHemis(wo, normal))
        return vec3(0);

    auto eta_inv = inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
    auto fdr_int = inside ? fdr_ext_ : fdr_int_;

    auto albedo = vec3(0);
    auto diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(texcoord) : vec3(0.5);
    if (nonlinear_)
        albedo = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int);
    else
        albedo = diffuse_reflectance / (1.0 - fdr_int);

    auto kr_i = Fresnel(wi, normal, eta_inv);
    auto kr_o = Fresnel(-wo, normal, eta_inv);
    albedo *= eta_inv * eta_inv * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;

    if (SameDirection(Reflect(wi, normal), wo))
    {
        auto attenuation = vec3(kr_i);
        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        albedo += attenuation;
    }

    return albedo;
}

__device__ Float Material::PdfPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    if (NotSameHemis(wo, normal))
        return 0;

    auto eta_inv = inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto kr = Fresnel(wi, normal, eta_inv);
    auto specular_sampling_weight = SpecularSamplingWeight(texcoord);

    auto pdf_specular = kr * specular_sampling_weight,
         pdf_diffuse = (1.0 - kr) * (1.0 - specular_sampling_weight);
    pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
    pdf_diffuse = 1.0 - pdf_specular;

    auto wo_local = ToLocal(wo, normal);
    auto local_pdf_diffuse = PdfHemisCos(wo_local);
    auto pdf = pdf_diffuse * local_pdf_diffuse;

    if (SameDirection(wo, Reflect(wi, normal)))
    {
        pdf += pdf_specular;
    }
    return pdf;
}
