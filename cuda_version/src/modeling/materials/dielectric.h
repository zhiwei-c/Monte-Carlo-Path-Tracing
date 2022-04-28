#pragma once

#include "material_base.h"

__global__ void InitDielectric(uint m_idx,
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

        auto specular_reflectance = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].specular_reflectance_idx != kUintMax)
            specular_reflectance = texture_list + material_info_list[m_idx].specular_reflectance_idx;

        auto specular_transmittance = static_cast<Texture *>(nullptr);
        if (material_info_list[m_idx].specular_transmittance_idx != kUintMax)
            specular_transmittance = texture_list + material_info_list[m_idx].specular_transmittance_idx;

        material_list[m_idx].InitDielectric(material_info_list[m_idx].twosided,
                                            bump_map,
                                            opacity_map,
                                            material_info_list[m_idx].eta,
                                            specular_reflectance,
                                            specular_transmittance);
    }
}

__device__ void Material::InitDielectric(bool twosided,
                                         Texture *bump_map,
                                         Texture *opacity_map,
                                         vec3 eta,
                                         Texture *specular_reflectance,
                                         Texture *specular_transmittance)
{
    type_ = kDielectric;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    eta_d_ = eta.x;
    eta_inv_d_ = 1.0 / eta.x;
    specular_reflectance_ = specular_reflectance;
    specular_transmittance_ = specular_transmittance;
}

__device__ void Material::SampleDielectric(BsdfSampling &bs, const vec3 &sample) const
{
    auto eta_inv = bs.inside ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto kr = Fresnel(-bs.wo, bs.normal, eta_inv);
    if (sample.x < kr)
        bs.wi = -Reflect(-bs.wo, bs.normal);
    else
        bs.wi = -Refract(-bs.wo, bs.normal, eta_inv);
    bs.pdf = PdfDielectric(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    if (bs.pdf < kEpsilonPdf)
        return;
    bs.attenuation = EvalDielectric(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    bs.valid = true;
}

__device__ vec3 Material::EvalDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto eta_inv = inside == kTrue ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto kr = Fresnel(wi, normal, eta_inv);
    if (SameDirection(wo, Reflect(wi, normal)))
    {
        auto attenuation = vec3(kr);
        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        return attenuation;
    }
    else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
    {
        auto attenuation = vec3(1.0 - kr);
        if (specular_transmittance_)
            attenuation *= specular_transmittance_->Color(texcoord);
        //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
        attenuation *= eta_inv * eta_inv;
        return attenuation;
    }
    else
        return vec3(0);
}

__device__ Float Material::PdfDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    auto eta_inv = inside == kTrue ? eta_d_ : eta_inv_d_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

    auto kr = Fresnel(wi, normal, eta_inv);
    if (SameDirection(wo, Reflect(wi, normal)))
        return kr;
    else if (kr > 1 - kEpsilon)
        return 0;
    else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
        return 1.0 - kr;
    else
        return 0;
}
