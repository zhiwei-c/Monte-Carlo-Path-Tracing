#pragma once

#include "material_base.h"

__device__ void Material::InitAreaLight(bool twosided, Texture *radiance)
{
    type_ = kAreaLight;
    radiance_ = radiance;
    twosided_ = twosided;
}

__global__ void InitAreaLight(uint m_idx,
                              MaterialInfo *material_info_list,
                              Texture *texture_list,
                              Material *material_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        material_list[m_idx].InitAreaLight(material_info_list[m_idx].twosided,
                                           texture_list + material_info_list[m_idx].radiance_idx);
    }
}