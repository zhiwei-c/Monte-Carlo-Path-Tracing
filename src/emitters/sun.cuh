#pragma once

#include "emitter.cuh"

class Sun : public Emitter
{
public:
    QUALIFIER_DEVICE Sun(const uint32_t id, const Emitter::Info::Data::Sun &data)
        : Emitter(id, Type::kSun), cos_cutoff_angle_(data.cos_cutoff_angle),
          direction_(data.direction), radiance_(data.radiance), id_texture_(data.id_texture)
    {
    }

    QUALIFIER_DEVICE bool GetRadiance(const Vec3 &origin, const Accel *accel, Bsdf **bsdf_buffer,
                                      Texture **texture_buffer, const float *pixel_buffer,
                                      uint32_t *seed, Vec3 *radiance, Vec3 *wi) const override;

    QUALIFIER_DEVICE Vec3 GetRadianceDirect(Vec3 look_dir, const float *pixel_buffer,
                                            Texture **texture_buffer) const;

private:
    float cos_cutoff_angle_;
    uint32_t id_texture_;
    Vec3 direction_;
    Vec3 radiance_;
};