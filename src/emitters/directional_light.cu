#include "directional_light.cuh"

QUALIFIER_DEVICE bool DirectionalLight::GetRadiance(const Vec3 &origin, const Accel *accel,
                                                    Bsdf **bsdf_buffer, Texture **texture_buffer,
                                                    const float *pixel_buffer, uint64_t *seed,
                                                    Vec3 *radiance, Vec3 *wi) const
{
    Intersection its;
    accel->Intersect(Ray(origin, -direction_), bsdf_buffer, texture_buffer, pixel_buffer, seed, &its);
    if (its.valid())
    {
        return false;
    }
    else
    {
        *radiance = radiance_;
        *wi = direction_;
        return true;
    }
}