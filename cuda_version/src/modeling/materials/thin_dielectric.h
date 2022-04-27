#pragma once

#include "material_base.h"

__global__ void InitThinDielectric(uint m_idx,
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

        material_list[m_idx].InitThinDielectric(material_info_list[m_idx].twosided,
                                                bump_map,
                                                opacity_map,
                                                material_info_list[m_idx].eta,
                                                specular_reflectance,
                                                specular_transmittance);
    }
}

__device__ void Material::InitThinDielectric(bool twosided,
                                             Texture *bump_map,
                                             Texture *opacity_map,
                                             vec3 eta,
                                             Texture *specular_reflectance,
                                             Texture *specular_transmittance)
{

    type_ = kThinDielectric;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    eta_d_ = eta.x;
    specular_reflectance_ = specular_reflectance;
    specular_transmittance_ = specular_transmittance;
}

__device__ void Material::SampleThinDielectric(BsdfSampling &bs, const vec3 &sample) const
{
    auto kr = Fresnel(-bs.wo, bs.normal, eta_d_);

    //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
    if (kr < 1)
        kr *= 2.0 / (1.0 + kr);

    if (sample.x < kr)
    {
        bs.pdf = kr;
        bs.wi = -Reflect(-bs.wo, bs.normal);
        bs.attenuation = vec3(kr);
        if (specular_reflectance_)
            bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
    }
    else
    {
        bs.pdf = 1.0 - kr;
        bs.wi = bs.wo;
        bs.attenuation = (1.0 - kr);
        if (specular_transmittance_)
            bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
    }
    bs.valid = (bs.pdf > kEpsilonPdf);
}
__device__ vec3 Material::EvalThinDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    auto kr = Fresnel(wi, normal, eta_d_);
    //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
    if (kr < 1)
        kr *= 2.0 / (1.0 + kr);

    if (SameDirection(wo, Reflect(wi, normal)))
    {
        auto attenuation = vec3(kr);
        if (specular_reflectance_)
            attenuation *= specular_reflectance_->Color(texcoord);
        return attenuation;
    }
    else if (SameDirection(wo, wi))
    {
        auto attenuation = vec3(1.0 - kr);
        if (specular_transmittance_)
            attenuation *= specular_transmittance_->Color(texcoord);
        return attenuation;
    }
    else
        return vec3(0);
}

__device__ Float Material::PdfThinDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, bool inside) const
{
    auto kr = Fresnel(wi, normal, eta_d_);
    //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
    if (kr < 1)
        kr *= 2.0 / (1.0 + kr);

    if (SameDirection(wo, Reflect(wi, normal)))
        return kr;
    else if (SameDirection(wo, wi))
        return 1.0 - kr;
    else
        return 0;
}
