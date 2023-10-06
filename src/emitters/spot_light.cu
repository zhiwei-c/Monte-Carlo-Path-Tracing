#include "spot_light.cuh"
#include "../utils/math.cuh"

QUALIFIER_DEVICE SpotLight::SpotLight(const uint64_t id, const Emitter::Info::Data::Spot &data)

    : Emitter(id, Emitter::Type::kSpot), intensity_(data.intensity),
      cutoff_angle_(data.cutoff_angle), cos_cutoff_angle_(cosf(data.cutoff_angle)),
      uv_factor_(tanf(data.cutoff_angle)), cos_beam_width_(cosf(data.beam_width)),
      transition_width_rcp_(1.0f / (data.cutoff_angle - data.beam_width)),
      id_texture_(data.id_texture), to_local_(data.to_world.Inverse()),
      position_world_(TransfromPoint(data.to_world, {0, 0, 0}))
{
}

QUALIFIER_DEVICE bool SpotLight::GetRadiance(const Vec3 &origin, const Accel *accel, Bsdf **bsdf_buffer,
                                             Texture **texture_buffer, const float *pixel_buffer,
                                             uint64_t *seed, Vec3 *radiance, Vec3 *wi) const
{
    const float distance = Length(position_world_ - origin);
    if (distance < kEpsilonFloat)
        return false;
    *wi = Normalize(origin - position_world_);
    Intersection its;
    accel->Intersect(Ray(origin, -*wi), bsdf_buffer, texture_buffer, pixel_buffer, seed, &its);
    if (its.valid() && its.distance() + kEpsilonDistance < distance)
        return false;

    const Vec3 dir_local = TransfromVector(to_local_, *wi);
    const float cos_theta = dir_local.z;
    if (cos_theta < cos_cutoff_angle_)
        return false;

    Vec3 fall_off = {1.0f, 1.0f, 1.0f};
    if (id_texture_ != kInvalidId)
    {
        const Vec2 texcoord = {0.5f + 0.5f * dir_local.x / (dir_local.z * uv_factor_),
                               0.5f + 0.5f * dir_local.y / (dir_local.z * uv_factor_)};
        fall_off *= texture_buffer[id_texture_]->GetColor(texcoord, pixel_buffer);
    }
    if (cos_theta < cos_beam_width_)
        fall_off *= (cutoff_angle_ - acosf(cos_theta)) * transition_width_rcp_;

    *radiance = intensity_ * fall_off * pow(1.0f / distance, 2);
    return true;
}
