#include "sun.cuh"

#include "../utils/math.cuh"

QUALIFIER_DEVICE bool Sun::GetRadiance(const Vec3 &origin, const Accel *accel, Bsdf **bsdf_buffer,
                                       Texture **texture_buffer, const float *pixel_buffer,
                                       uint64_t *seed, Vec3 *radiance, Vec3 *wi) const
{
    const Vec3 dir = SampleConeUniform(cos_cutoff_angle_, RandomFloat(seed), RandomFloat(seed));
    *wi = ToWorld(dir, direction_);

    Intersection its;
    accel->Intersect(Ray(origin, -*wi), bsdf_buffer, texture_buffer, pixel_buffer, seed, &its);
    if (its.valid())
    {
        return false;
    }
    else
    {
        *radiance = radiance_;
        return true;
    }
}

QUALIFIER_DEVICE Vec3 Sun::GetRadianceDirect(Vec3 look_dir, const float *pixel_buffer,
                                             Texture **texture_buffer) const
{
    float phi = 0, theta = 0;
    CartesianToSpherical(look_dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * kOneDivTwoPi, theta * kPiInv};
    return texture_buffer[id_texture_]->GetColor(texcoord, pixel_buffer);
}