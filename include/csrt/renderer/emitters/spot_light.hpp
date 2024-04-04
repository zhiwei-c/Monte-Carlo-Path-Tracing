#ifndef CSRT__RENDERER__EMITTERS__SPOT_LIGHT_HPP
#define CSRT__RENDERER__EMITTERS__SPOT_LIGHT_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct EmitterSampleRec;

struct SpotLightInfo
{
    float cutoff_angle = 0;
    float beam_width = 0;
    uint32_t id_texture = kInvalidId;
    Vec3 intensity = {};
    Mat4 to_world = {};
};

struct SpotLightData
{
    float cutoff_angle = 0;
    float cos_cutoff_angle = 0;
    float uv_factor = 0;
    float beam_width = 0;
    float cos_beam_width = 0;
    float transition_width_rcp = 0;
    Texture *texture = nullptr;
    Vec3 intensity = {};
    Vec3 position = {};
    Mat4 to_local = {};
};

QUALIFIER_D_H void SampleSpotLight(const SpotLightData &data,
                                   const Vec3 &origin, const float xi_0,
                                   const float xi_1, EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateSpotLight(const SpotLightData &data,
                                     const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateSpotLight(const SpotLightData &data,
                                     const Vec3 &look_dir);

QUALIFIER_D_H float PdfSpotLight(const SpotLightData &data,
                                 const Vec3 &look_dir);

} // namespace csrt

#endif