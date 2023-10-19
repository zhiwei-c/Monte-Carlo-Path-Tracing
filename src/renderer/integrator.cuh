#pragma once

#include "intersection.cuh"
#include "ray.cuh"
#include "../global.cuh"
#include "../accelerators/accel.cuh"
#include "../emitters/emitters.cuh"
#include "../textures/texture.cuh"
#include "../geometry/primitive.cuh"
#include "../geometry/instance.cuh"

class Integrator
{
public:
    QUALIFIER_DEVICE Integrator(float *pixel_buffer, Texture **texture_buffer, Bsdf **bsdf_buffer,
                                Primitive *primitive_buffer, Instance *instance_buffer,
                                Accel *accel, uint32_t num_emitter, Emitter **emitter_buffer,
                                uint32_t num_area_light, uint32_t *area_light_id_buffer,
                                EnvMap *env_map, Sun *sun);

    QUALIFIER_DEVICE Vec3 GenerateRay(const Vec3 &eye, const Vec3 &look_dir, uint32_t *seed) const;

private:
    QUALIFIER_DEVICE SamplingRecord SampleRay(const Vec3 &wo, const Intersection &its, Bsdf **bsdf,
                                              uint32_t *seed) const;
    QUALIFIER_DEVICE SamplingRecord EvaluateRay(const Vec3 &wi, const Vec3 &wo,
                                                const Intersection &its, Bsdf **bsdf, 
                                                uint32_t *seed) const;

    QUALIFIER_DEVICE Vec3 EvaluateDirectAreaLight(const Intersection &its, const Vec3 &wo,
                                                  uint32_t *seed) const;
    QUALIFIER_DEVICE Vec3 EvaluateDirectOtherLight(const Intersection &its, const Vec3 &wo,
                                                   uint32_t *seed) const;
    QUALIFIER_DEVICE float PdfDirectLight(const Intersection &its_pre, const Vec3 &wi,
                                          const float cos_theta_prime) const;

    uint32_t num_emitter_;
    uint32_t num_area_light_;
    const float *pixel_buffer_;
    Texture **texture_buffer_;
    Bsdf **bsdf_buffer_;
    Primitive *primitive_buffer_;
    const Instance *instance_buffer_;
    const Accel *accel_;
    const uint32_t *area_light_id_buffer_;
    Emitter **emitter_buffer_;
    EnvMap *env_map_;
    Sun *sun_;
};