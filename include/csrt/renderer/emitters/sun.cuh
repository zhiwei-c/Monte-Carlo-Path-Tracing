#pragma once

#include "../../rtcore/scene.cuh"
#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

namespace csrt
{

struct EmitterSampleRec;

struct SunInfo
{
    float cos_cutoff_angle = 0;
    uint32_t id_texture = kInvalidId;
    Vec3 direction = {};
    Vec3 radiance = {};
};

struct SunData
{
    float cos_cutoff_angle = 0;
    Texture *texture = nullptr;
    Vec3 direction = {};
    Vec3 radiance = {};
};

QUALIFIER_D_H void SampleSun(const SunData &data, const Vec3 &origin,
                             const float xi_0, const float xi_1,
                             EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateSun(const SunData &data,
                               const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateSun(const SunData &data, const Vec3 &look_dir);

QUALIFIER_D_H float PdfSun(const SunData &data, const Vec3 &look_dir);

} // namespace csrt