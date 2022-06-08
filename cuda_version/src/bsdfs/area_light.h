#pragma once

#include "../core/bsdf.h"

class AreaLight : public Bsdf
{
public:
    __device__ AreaLight(uint idx, bool twosided, Texture *radiance)
        : Bsdf(idx, kAreaLight, twosided, nullptr, nullptr), radiance_(radiance)
    {
    }

    __device__ vec3 radiance() const override
    {
        return radiance_->Color(vec2(0));
    }

private:
    Texture *radiance_;
};

__device__ inline void InitAreaLight(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
                                     Bsdf **&bsdf_list)
{
    auto radiance = texture_list + bsdf_info_list[m_idx].radiance_idx;
    bsdf_list[m_idx] = new AreaLight(m_idx, bsdf_info_list[m_idx].twosided, radiance);
}