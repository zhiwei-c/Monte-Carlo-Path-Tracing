#ifndef CSRT__RENDERER__BSDF__DIFFUSE_HPP
#define CSRT__RENDERER__BSDF__DIFFUSE_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

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

#endif