#ifndef CSRT__RENDERER__EMITTERS__POINT_LIGHT_HPP
#define CSRT__RENDERER__EMITTERS__POINT_LIGHT_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct EmitterSampleRec;

struct PointLightData
{
    Vec3 position = {};
    Vec3 intensity = {};
};

QUALIFIER_D_H void SamplePointLight(const PointLightData &data,
                                    const Vec3 &origin, const float xi_0,
                                    const float xi_1, EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluatePointLight(const PointLightData &data,
                                      const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluatePointLight(const PointLightData &data,
                                      const Vec3 &look_dir);

QUALIFIER_D_H float PdfPointLight(const PointLightData &data,
                                  const Vec3 &look_dir);

} // namespace csrt

#endif