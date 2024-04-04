#ifndef CSRT__RENDERER__BSDF__PLASTIC_HPP
#define CSRT__RENDERER__BSDF__PLASTIC_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct BsdfSampleRec;

struct PlasticInfo
{
    float eta = 1.0f;
    uint32_t id_roughness = kInvalidId;
    uint32_t id_diffuse_reflectance = kInvalidId;
    uint32_t id_specular_reflectance = kInvalidId;
};

struct PlasticData
{
    float reflectivity = 1.0f;
    float F_avg = 1.0f;
    Texture *roughness = nullptr;
    Texture *diffuse_reflectance = nullptr;
    Texture *specular_reflectance = nullptr;
};

QUALIFIER_D_H void EvaluatePlastic(const PlasticData &data, BsdfSampleRec *rec);

QUALIFIER_D_H void SamplePlastic(const PlasticData &data, uint32_t *seed,
                                 BsdfSampleRec *rec);

} // namespace csrt

#endif