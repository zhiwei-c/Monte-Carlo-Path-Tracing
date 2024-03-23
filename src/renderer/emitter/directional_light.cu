#include "csrt/renderer/emitters/directional_light.cuh"

#include "csrt/renderer/emitters/emitter.cuh"

namespace csrt
{

QUALIFIER_D_H void SampleDirectionalLight(const DirectionalLightData &data,
                                          const Vec3 &origin, const float xi_0,
                                          const float xi_1,
                                          EmitterSampleRec *rec)
{
    *rec = {
        true,          // valid
        true,          // harsh
        kMaxFloat,     // distance
        data.direction // wi
    };
}

QUALIFIER_D_H Vec3 EvaluateDirectionalLight(const DirectionalLightData &data,
                                            const EmitterSampleRec *rec)
{
    return data.radiance;
}

QUALIFIER_D_H Vec3 EvaluateDirectionalLight(const DirectionalLightData &data,
                                            const Vec3 &look_dir)
{
    return {};
}

QUALIFIER_D_H float PdfDirectionalLight(const DirectionalLightData &data,
                                        const Vec3 &look_dir)
{
    return 0;
}

} // namespace csrt