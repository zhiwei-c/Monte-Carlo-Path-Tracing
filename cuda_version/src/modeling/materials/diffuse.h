#pragma once

#include "material_base.h"

__device__ void Material::InitDiffuse(bool twosided,
                                      Texture *bump_map,
                                      Texture *opacity_map,
                                      Texture *diffuse_reflectance)
{
    type_ = kDiffuse;
    twosided_ = twosided;
    bump_map_ = bump_map;
    opacity_map_ = opacity_map;
    diffuse_reflectance_ = diffuse_reflectance;
}

__device__ void Material::SampleDiffuse(BsdfSampling &bs, const vec3 &sample) const
{
    auto wi_local = vec3(0);
    Float pdf = 0;
    HemisCos(sample.x, sample.y, wi_local, pdf);
    if (pdf < kEpsilonPdf)
        return;

    bs.pdf = pdf;
    bs.wi = -ToWorld(wi_local, bs.normal);

    if (diffuse_reflectance_ != nullptr)
        bs.attenuation = diffuse_reflectance_->Color(bs.texcoord) * kPiInv;
    else
        bs.attenuation = vec3(0.5 * kPiInv);
    bs.valid = true;
}

__device__ vec3 Material::EvalDiffuse(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    if (NotSameHemis(wo, normal))
        return vec3(0);
    else
    {
        if (diffuse_reflectance_ != nullptr)
            return diffuse_reflectance_->Color(texcoord) * kPiInv;
        else
            return vec3(0.5 * kPiInv);
    }
}

__device__ Float Material::PdfDiffuse(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const
{
    if (NotSameHemis(wo, normal))
        return 0;

    auto wo_local = ToLocal(wo, normal);
    auto pdf = PdfHemisCos(wo_local);
    return pdf;
}

__device__ inline  void InitDiffuse(uint m_idx,
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

    auto diffuse_reflectance = static_cast<Texture *>(nullptr);
    if (material_info_list[m_idx].diffuse_reflectance_idx != kUintMax)
        diffuse_reflectance = texture_list + material_info_list[m_idx].diffuse_reflectance_idx;

    material_list[m_idx].InitDiffuse(material_info_list[m_idx].twosided,
                                     bump_map,
                                     opacity_map,
                                     diffuse_reflectance);
}