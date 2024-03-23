#include "csrt/renderer/emitters/sun.cuh"

#include "csrt/renderer/emitters/emitter.cuh"

namespace csrt
{

QUALIFIER_D_H void SampleSun(const SunData &data, const Vec3 &origin,
                             const float xi_0, const float xi_1,
                             EmitterSampleRec *rec)
{
    const Vec3 dir_local = SampleConeUniform(data.cos_cutoff_angle, xi_0, xi_1);
    *rec = {
        true,                                   // valid
        true,                                   // harsh
        kMaxFloat,                              // distance
        LocalToWorld(dir_local, data.direction) // wi
    };
}

QUALIFIER_D_H Vec3 EvaluateSun(const SunData &data, const EmitterSampleRec *rec)
{
    return data.radiance;
}

QUALIFIER_D_H Vec3 EvaluateSun(const SunData &data, const Vec3 &look_dir)
{
    float phi = 0, theta = 0;
    CartesianToSpherical(look_dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data.texture->GetColor(texcoord);
}

QUALIFIER_D_H float PdfSun(const SunData &data, const Vec3 &look_dir)
{
    return 0;
}

} // namespace csrt