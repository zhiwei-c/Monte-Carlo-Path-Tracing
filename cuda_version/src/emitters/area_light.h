#pragma once

#include "../core/material_base.h"

__device__ void Material::InitAreaLight(bool twosided, Texture *radiance)
{
    type_ = kAreaLight;
    radiance_ = radiance;
    twosided_ = twosided;
}

__device__ inline void InitAreaLight(uint m_idx,
                                     MaterialInfo *material_info_list,
                                     Texture *texture_list,
                                     Material *material_list)
{
    auto radiance = texture_list + material_info_list[m_idx].radiance_idx;
    material_list[m_idx].InitAreaLight(material_info_list[m_idx].twosided, radiance);
}