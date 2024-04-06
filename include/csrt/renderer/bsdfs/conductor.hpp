#ifndef CSRT__RENDERER__BSDF__CONDUCTOR_HPP
#define CSRT__RENDERER__BSDF__CONDUCTOR_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

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
    Vec3 F_avg = {};
    float *brdf_avg_buffer = nullptr;
    float *albedo_avg_buffer = nullptr;
};

QUALIFIER_D_H void SampleConductor(const ConductorData &data, uint32_t *seed,
                                   BsdfSampleRec *rec);

QUALIFIER_D_H void EvaluateConductor(const ConductorData &data,
                                     BsdfSampleRec *rec);

} // namespace csrt

#endif