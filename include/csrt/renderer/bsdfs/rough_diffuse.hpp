#ifndef CSRT__RENDERER__BSDF__ROUGH_PLASTIC_HPP
#define CSRT__RENDERER__BSDF__ROUGH_PLASTIC_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct BsdfSampleRec;

struct RoughDiffuseInfo
{
    bool use_fast_approx = true;
    uint32_t id_diffuse_reflectance = kInvalidId;
    uint32_t id_roughness = kInvalidId;
};

struct RoughDiffuseData
{
    bool use_fast_approx = true;
    Texture *diffuse_reflectance = nullptr;
    Texture *roughness = nullptr;
};

QUALIFIER_D_H void EvaluateRoughDiffuse(const RoughDiffuseData &data,
                                        BsdfSampleRec *rec);

QUALIFIER_D_H void SampleRoughDiffuse(const RoughDiffuseData &data,
                                      uint32_t *seed, BsdfSampleRec *rec);

} // namespace csrt

#endif