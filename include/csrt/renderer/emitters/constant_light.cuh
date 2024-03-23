#pragma once

#include "../../rtcore/scene.cuh"
#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

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