#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

namespace csrt
{
struct BsdfSampleRec;

struct ConductorInfo
{
    uint32_t id_roughness_u = kInvalidId;
    uint32_t id_roughness_v = kInvalidId;
    uint32_t id_specular_reflectance = kInvalidId;
    Vec3 reflectivity = {};
    Vec3 edgetint = {};
};

struct ConductorData
{
    Texture *roughness_u = nullptr;
    Texture *roughness_v = nullptr;
    Texture *specular_reflectance = nullptr;
    Vec3 reflectivity = {};
    Vec3 edgetint = {};
};

QUALIFIER_D_H void EvaluateConductor(const ConductorData &data,
                                     BsdfSampleRec *rec);

QUALIFIER_D_H void SampleConductor(const ConductorData &data, uint32_t *seed,
                                   BsdfSampleRec *rec);

} // namespace csrt