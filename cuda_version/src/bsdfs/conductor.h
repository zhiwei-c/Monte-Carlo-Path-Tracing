#pragma once

#include "../core/material_base.h"

class Conductor : public Material
{
public:
    __device__ Conductor(uint idx,
                         bool twosided,
                         Texture *bump_map,
                         Texture *opacity_map,
                         bool mirror,
                         vec3 eta,
                         vec3 k,
                         Texture *specular_reflectance)
        : Material(idx, kConductor, twosided, bump_map, opacity_map),
          mirror_(mirror),
          eta_(eta),
          k_(k),
          specular_reflectance_(specular_reflectance) {}

    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const override
    {
        bs.pdf = 1;
        bs.wi = -Reflect(-bs.wo, bs.normal);

        bs.attenuation = vec3(1);
        if (specular_reflectance_)
            bs.attenuation = specular_reflectance_->Color(bs.texcoord);
        if (!mirror_)
            bs.attenuation *= FresnelConductor(bs.wi, bs.normal, eta_, k_);

        bs.valid = true;
    }

    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        if (!SameDirection(wo, Reflect(wi, normal)))
            return vec3(0);

        auto albedo = vec3(1);
        if (specular_reflectance_)
            albedo = specular_reflectance_->Color(texcoord);
        if (!mirror_)
            albedo *= FresnelConductor(wi, normal, eta_, k_);

        return albedo;
    }

    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        if (SameDirection(wo, Reflect(wi, normal)))
            return 1;
        else
            return 0;
    }

private:
    bool mirror_;
    vec3 eta_;
    vec3 k_;
    Texture *specular_reflectance_;
};

__device__ inline void InitConductor(size_t m_idx,
                                     MaterialInfo *material_info_list,
                                     Texture *texture_list,
                                     Material **&material_list)
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

    material_list[m_idx] = new Conductor(m_idx,
                                         material_info_list[m_idx].twosided,
                                         bump_map,
                                         opacity_map,
                                         material_info_list[m_idx].mirror,
                                         material_info_list[m_idx].eta,
                                         material_info_list[m_idx].k,
                                         specular_reflectance);
}