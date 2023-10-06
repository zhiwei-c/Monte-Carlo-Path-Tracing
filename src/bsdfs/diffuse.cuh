#pragma once

#include "bsdf.cuh"

class Diffuse : public Bsdf
{
public:
    QUALIFIER_DEVICE Diffuse(const uint64_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kDiffuse, data.twosided, data.id_opacity, data.id_bumpmap),
          id_diffuse_reflectance_(data.diffuse.id_diffuse_reflectance)
    {
    }

    QUALIFIER_DEVICE void Evaluate(const float *pixel_buffer, Texture **texture_buffer,
                                   uint64_t *seed, SamplingRecord *rec) const override;

    QUALIFIER_DEVICE void Sample(const float *pixel_buffer, Texture **texture_buffer,
                                 uint64_t *seed, SamplingRecord *rec) const override;

private:
    uint64_t id_diffuse_reflectance_;
};