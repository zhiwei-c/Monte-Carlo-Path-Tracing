#include "csrt/renderer/emitters/spot_light.cuh"

#include "csrt/renderer/emitters/emitter.cuh"

namespace csrt
{

QUALIFIER_D_H void SampleSpotLight(const SpotLightData &data,
                                   const Vec3 &origin, const float xi_0,
                                   const float xi_1, EmitterSampleRec *rec)
{
    const Vec3 vec = origin - data.position;
    const Vec3 wi = Normalize(vec),
               dir_local = TransformVector(data.to_local, wi);
    if (dir_local.z >= data.cos_cutoff_angle)
    {
        *rec = {
            true,        // valid
            true,        // harsh
            Length(vec), // distance
            wi           // wi
        };
    }
}

QUALIFIER_D_H Vec3 EvaluateSpotLight(const SpotLightData &data,
                                     const EmitterSampleRec *rec)
{
    const Vec3 dir = TransformVector(data.to_local, rec->wi);

    Vec3 fall_off = {1.0f, 1.0f, 1.0f};
    if (data.texture != nullptr)
    {
        const Vec2 texcoord = {0.5f + 0.5f * dir.x / (dir.z * data.uv_factor),
                               0.5f + 0.5f * dir.y / (dir.z * data.uv_factor)};
        fall_off *= data.texture->GetColor(texcoord);
    }
    if (dir.z < data.cos_beam_width)
    {
        fall_off *=
            (data.cutoff_angle - acosf(dir.z)) * data.transition_width_rcp;
    }
    return data.intensity * fall_off * Sqr(1.0f / rec->distance);
}

QUALIFIER_D_H Vec3 EvaluateSpotLight(const SpotLightData &data,
                                     const Vec3 &look_dir)
{
    return {};
}

QUALIFIER_D_H float PdfSpotLight(const SpotLightData &data,
                                 const Vec3 &look_dir)
{
    return 0;
}

} // namespace csrt