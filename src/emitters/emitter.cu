#include "emitter.cuh"

#include "../utils/math.cuh"

namespace
{
    constexpr float kSunAppRadius = 0.5358;
} // namespace

Emitter::Info Emitter::Info::CreateSpotLight(const Mat4 &to_world, const Vec3 &intensity,
                                             const float cutoff_angle, const float beam_width,
                                             const uint32_t id_texture)
{
    Info info;
    info.type = Emitter::Type::kSpot;
    info.data.spot.to_world = to_world;
    info.data.spot.intensity = intensity;
    info.data.spot.cutoff_angle = cutoff_angle;
    info.data.spot.beam_width = beam_width;
    info.data.spot.id_texture = id_texture;
    return info;
}

Emitter::Info Emitter::Info::CreateSun(const Vec3 &direction, const Vec3 &radiance,
                                       const float radius_scale, uint32_t id_texture)
{
    Info info;
    info.type = Emitter::Type::kSun;
    info.data.sun.cos_cutoff_angle = cosf(ToRadians(kSunAppRadius * 0.5f) * radius_scale);
    info.data.sun.direction = direction;
    info.data.sun.id_texture = id_texture;
    info.data.sun.radiance = radiance;
    return info;
}

Emitter::Info Emitter::Info::CreateDirctional(const Vec3 &direction, const Vec3 &radiance)
{
    Info info;
    info.type = Emitter::Type::kDirectional;
    info.data.directional.direction = direction;
    info.data.directional.radiance = radiance;
    return info;
}
