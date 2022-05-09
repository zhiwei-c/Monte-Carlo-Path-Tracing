#pragma once

#include "../core/material_base.h"

class AreaLight : public Material
{
public:
    __device__ AreaLight(uint idx,
                         bool twosided,
                         Texture *radiance)
        : Material(idx, kAreaLight,twosided, nullptr, nullptr),
          radiance_(radiance) {}

    __device__ vec3 radiance() const override
    {
        return radiance_->Color(vec2(0));
    }

private:
    Texture *radiance_;
};

__device__ inline void InitAreaLight(uint m_idx,
                                     MaterialInfo *material_info_list,
                                     Texture *texture_list,
                                     Material **&material_list)
{
    auto radiance = texture_list + material_info_list[m_idx].radiance_idx;
    material_list[m_idx] = new AreaLight(m_idx,
                                         material_info_list[m_idx].twosided,
                                         radiance);
}