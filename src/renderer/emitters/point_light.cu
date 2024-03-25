#include "csrt/renderer/emitters/point_light.cuh"

#include "csrt/renderer/emitters/emitter.cuh"

namespace csrt
{

QUALIFIER_D_H void SamplePointLight(const PointLightData &data,
                                    const Vec3 &origin, const float xi_0,
                                    const float xi_1, EmitterSampleRec *rec)
{
    const Vec3 vec = origin - data.position;
    *rec = {
        true,          // valid
        true,          // harsh
        Length(vec),   // distance
        Normalize(vec) // wi
    };
}

QUALIFIER_D_H Vec3 EvaluatePointLight(const PointLightData &data,
                                      const EmitterSampleRec *rec)
{
    return {};
}

QUALIFIER_D_H Vec3 EvaluatePointLight(const PointLightData &data,
                                      const Vec3 &look_dir)
{
    return {};
}

QUALIFIER_D_H float PdfPointLight(const PointLightData &data,
                                  const Vec3 &look_dir)
{
    return 0;
}

} // namespace csrt