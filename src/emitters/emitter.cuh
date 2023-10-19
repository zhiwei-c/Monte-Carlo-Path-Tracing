#pragma once

#include "../tensor/tensor.cuh"
#include "../textures/texture.cuh"
#include "../accelerators/accel.cuh"

class Emitter
{
public:
    enum Type
    {
        kDirectional,
        kSpot,
        kSun,
    };

    struct Info
    {
        Type type;
        struct Data
        {
            struct Directional
            {
                Vec3 direction;
                Vec3 radiance;
            } directional;
            struct Spot
            {
                float cutoff_angle;
                float beam_width;
                uint32_t id_texture;
                Vec3 intensity;
                Mat4 to_world;
            } spot;
            struct Sun
            {
                float cos_cutoff_angle;
                uint32_t id_texture;
                Vec3 direction;
                Vec3 radiance;
            } sun;
        } data;

        static Info CreateDirctional(const Vec3 &direction, const Vec3 &radiance);
        static Info CreateSpotLight(const Mat4 &to_world, const Vec3 &intensity,
                                    const float cutoff_angle, const float beam_width,
                                    const uint32_t id_texture);
        static Info CreateSun(const Vec3 &direction, const Vec3 &radiance, const float radius_scale,
                              uint32_t id_texture);
    };

    QUALIFIER_DEVICE virtual ~Emitter() {}

    QUALIFIER_DEVICE virtual bool GetRadiance(const Vec3 &origin, const Accel *accel,
                                              Bsdf **bsdf_buffer, Texture **texture_buffer,
                                              const float *pixel_buffer, uint32_t *seed,
                                              Vec3 *radiance, Vec3 *wi) const = 0;

protected:
    QUALIFIER_DEVICE Emitter(const uint32_t id, const Type type)
        : id_(id), type_(type) {}

private:
    uint32_t id_;
    Type type_;
};