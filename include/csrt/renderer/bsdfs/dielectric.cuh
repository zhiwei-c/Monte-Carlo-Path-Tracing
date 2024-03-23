#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

namespace csrt
{
struct BsdfSampleRec;

struct DielectricInfo
{
    uint32_t id_roughness_u = kInvalidId;
    uint32_t id_roughness_v = kInvalidId;
    uint32_t id_specular_reflectance = kInvalidId;
    uint32_t id_specular_transmittance = kInvalidId;
    float eta = 1.0f;
};

struct DielectricData
{
    float reflectivity = 1.0f;
    float eta = 1.0f;
    float eta_inv = 1.0f;
    Texture *roughness_u = nullptr;
    Texture *roughness_v = nullptr;
    Texture *specular_reflectance = nullptr;
    Texture *specular_transmittance = nullptr;
};

QUALIFIER_D_H void EvaluateDielectric(const DielectricData &data,
                                      BsdfSampleRec *rec);

QUALIFIER_D_H void SampleDielectric(const DielectricData &data, uint32_t *seed,
                                    BsdfSampleRec *rec);

} // namespace csrt