#pragma once

#include "../../rtcore/scene.cuh"
#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"

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