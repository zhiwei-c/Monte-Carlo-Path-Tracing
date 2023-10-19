#pragma once

#include "bsdf.cuh"

class RoughDiffuse : public Bsdf
{
public:
    QUALIFIER_DEVICE RoughDiffuse(const uint32_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kRoughDiffuse, data.twosided, data.id_opacity, data.id_bumpmap),
          id_diffuse_reflectance_(data.rough_diffuse.id_diffuse_reflectance),
          id_roughness_(data.rough_diffuse.id_roughness),
          use_fast_approx_(data.rough_diffuse.use_fast_approx)
    {
    }

    QUALIFIER_DEVICE void Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                   uint32_t *seed, SamplingRecord *rec) const override;

    QUALIFIER_DEVICE void Sample(Texture **texture_buffer, const float *pixel_buffer,
                                 uint32_t *seed, SamplingRecord *rec) const override;

private:
    bool use_fast_approx_;
    uint32_t id_roughness_;
    uint32_t id_diffuse_reflectance_;
};