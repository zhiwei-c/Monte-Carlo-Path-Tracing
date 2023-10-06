#pragma once

#include "emitter.cuh"

class DirectionalLight : public Emitter
{
public:
    QUALIFIER_DEVICE DirectionalLight(const uint64_t id, const Emitter::Info::Data::Directional &data)
        : Emitter(id, Type::kDirectional), direction_(data.direction), radiance_(data.radiance)
    {
    }

    QUALIFIER_DEVICE bool GetRadiance(const Vec3 &origin, const Accel *accel, Bsdf **bsdf_buffer,
                                      Texture **texture_buffer, const float *pixel_buffer,
                                      uint64_t *seed, Vec3 *radiance, Vec3 *wi) const override;

private:
    Vec3 direction_; // 世界空间下的方向
    Vec3 radiance_;  // 辐射强度
};