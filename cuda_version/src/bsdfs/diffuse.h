#pragma once

#include "../core/material_base.h"

class Diffuse : public Material
{
public:
    __device__ Diffuse(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                       Texture *diffuse_reflectance)
        : Material(idx, kDiffuse, twosided, bump_map, opacity_map),
          diffuse_reflectance_(diffuse_reflectance) {}

    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const override
    {
        auto wi_local = vec3(0);
        auto pdf = static_cast<Float>(0);
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

    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        if (diffuse_reflectance_ != nullptr)
            return diffuse_reflectance_->Color(texcoord) * kPiInv;
        else
            return vec3(0.5 * kPiInv);
    }

    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        auto wo_local = ToLocal(wo, normal);
        auto pdf = PdfHemisCos(wo_local);
        return pdf;
    }

    __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const override
    {
        if (Material::Transparent(texcoord, sample))
            return true;
        if (diffuse_reflectance_)
        {
            if (diffuse_reflectance_->type() == kBitmap &&
                diffuse_reflectance_->Transparent(texcoord, sample.y))
                return true;
        }
        return false;
    }

private:
    Texture *diffuse_reflectance_;
};

__device__ inline void InitDiffuse(uint m_idx,
                                   MaterialInfo *material_info_list,
                                   Texture *texture_list,
                                   Material **&material_list)
{
    Texture * bump_map = nullptr;
    if (material_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + material_info_list[m_idx].bump_map_idx;

    Texture * opacity_map = nullptr;
    if (material_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + material_info_list[m_idx].opacity_idx;

    Texture * diffuse_reflectance = nullptr;
    if (material_info_list[m_idx].diffuse_reflectance_idx != kUintMax)
        diffuse_reflectance = texture_list + material_info_list[m_idx].diffuse_reflectance_idx;

    material_list[m_idx] = new Diffuse(m_idx,
                                       material_info_list[m_idx].twosided,
                                       bump_map,
                                       opacity_map,
                                       diffuse_reflectance);
}