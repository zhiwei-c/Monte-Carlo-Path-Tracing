#ifndef CSRT__RENDERER__EMITTERS__CONSTANT_LIGHT_HPP
#define CSRT__RENDERER__EMITTERS__CONSTANT_LIGHT_HPP

#include "../../rtcore/scene.hpp"
#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"

namespace csrt
{

struct EmitterSampleRec;

struct ConstantLightData
{
    Vec3 radiance = {};
};

QUALIFIER_D_H void SampleConstantLight(const ConstantLightData &data,
                                       const Vec3 &origin, const float xi_0,
                                       const float xi_1, EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateConstantLight(const ConstantLightData &data,
                                         const EmitterSampleRec *rec);

QUALIFIER_D_H Vec3 EvaluateConstantLight(const ConstantLightData &data,
                                         const Vec3 &look_dir);

QUALIFIER_D_H float PdfConstantLight(const ConstantLightData &data,
                                     const Vec3 &look_dir);

} // namespace csrt

#endif