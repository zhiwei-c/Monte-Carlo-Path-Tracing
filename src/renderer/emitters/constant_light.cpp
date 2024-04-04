#include "csrt/renderer/emitters/constant_light.hpp"

#include "csrt/renderer/emitters/emitter.hpp"

namespace csrt
{

QUALIFIER_D_H void SampleConstantLight(const ConstantLightData &data,
                                       const Vec3 &origin, const float xi_0,
                                       const float xi_1, EmitterSampleRec *rec)
{
    *rec = {
        true,                           // valid
        false,                          // harsh
        kMaxFloat,                      // distance
        SampleSphereUniform(xi_0, xi_1) // wi
    };
}

QUALIFIER_D_H Vec3 EvaluateConstantLight(const ConstantLightData &data,
                                         const EmitterSampleRec *rec)
{
    return data.radiance;
}

QUALIFIER_D_H Vec3 EvaluateConstantLight(const ConstantLightData &data,
                                         const Vec3 &look_dir)
{
    return data.radiance;
}

QUALIFIER_D_H float PdfConstantLight(const ConstantLightData &data,
                                     const Vec3 &look_dir)
{
    return k1Div4Pi;
}

} // namespace csrt