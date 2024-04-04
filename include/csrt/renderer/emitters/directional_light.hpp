#ifndef CSRT__RENDERER__EMITTERS__DIRECTIONAL_LIGHT_HPP
#define CSRT__RENDERER__EMITTERS__DIRECTIONAL_LIGHT_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct EmitterSampleRec;

struct DirectionalLightData
{
    Vec3 direction = {};
    Vec3 radiance = {};
};

QUALIFIER_D_H void SampleDirectionalLight(const DirectionalLightData &data,
                                          const Vec3 &origin, const float xi_0,
                                          const float xi_1,
                                          EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateDirectionalLight(const DirectionalLightData &data,
                                            const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateDirectionalLight(const DirectionalLightData &data,
                                            const Vec3 &look_dir);

QUALIFIER_D_H float PdfDirectionalLight(const DirectionalLightData &data,
                                        const Vec3 &look_dir);

} // namespace csrt

#endif