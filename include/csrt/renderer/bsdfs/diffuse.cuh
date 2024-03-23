#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

namespace csrt
{
struct BsdfSampleRec;

struct DiffuseInfo
{
    uint32_t id_diffuse_reflectance = kInvalidId;
};

struct DiffuseData
{
    Texture *diffuse_reflectance = nullptr;
};

QUALIFIER_D_H void EvaluateDiffuse(const DiffuseData &data, BsdfSampleRec *rec);

QUALIFIER_D_H void SampleDiffuse(const DiffuseData &data, uint32_t *seed,
                                 BsdfSampleRec *rec);

} // namespace csrt