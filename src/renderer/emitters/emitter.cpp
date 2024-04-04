#include "csrt/renderer/emitters/emitter.hpp"

namespace csrt
{

QUALIFIER_D_H EmitterInfo::EmitterInfo()
    : type(EmitterType::kNone), point{}
{
}

QUALIFIER_D_H EmitterInfo::EmitterInfo(const EmitterInfo &info)
    : type(info.type)
{
    switch (info.type)
    {
    case EmitterType::kPoint:
        point = info.point;
        break;
    case EmitterType::kSpot:
        spot = info.spot;
        break;
    case EmitterType::kDirectional:
        directional = info.directional;
        break;
    case EmitterType::kSun:
        sun = info.sun;
        break;
    case EmitterType::kEnvMap:
        envmap = info.envmap;
        break;
    case EmitterType::kConstant:
        constant = info.constant;
        break;
    }
}

// QUALIFIER_D_H void EmitterInfo::operator=(const EmitterInfo &info)
// {
//     type = info.type;
//     switch (info.type)
//     {
//     case EmitterType::kPoint:
//         point = info.point;
//         break;
//     case EmitterType::kSpot:
//         spot = info.spot;
//         break;
//     case EmitterType::kDirectional:
//         directional = info.directional;
//         break;
//     case EmitterType::kSun:
//         sun = info.sun;
//         break;
//     case EmitterType::kEnvMap:
//         envmap = info.envmap;
//         break;
//     case EmitterType::kConstant:
//         constant = info.constant;
//         break;
//     }
// }

QUALIFIER_D_H EmitterData::EmitterData()
    : type(EmitterType::kNone), point{}
{
}

// QUALIFIER_D_H EmitterData::EmitterData(const EmitterData &info)
//     : type(info.type)
// {
//     switch (info.type)
//     {
//     case EmitterType::kPoint:
//         point = info.point;
//         break;
//     case EmitterType::kSpot:
//         spot = info.spot;
//         break;
//     case EmitterType::kDirectional:
//         directional = info.directional;
//         break;
//     case EmitterType::kSun:
//         sun = info.sun;
//         break;
//     case EmitterType::kEnvMap:
//         envmap = info.envmap;
//         break;
//     case EmitterType::kConstant:
//         constant = info.constant;
//         break;
//     }
// }

QUALIFIER_D_H void EmitterData::operator=(const EmitterData &data)
{
    type = data.type;
    switch (data.type)
    {
    case EmitterType::kPoint:
        point = data.point;
        break;
    case EmitterType::kSpot:
        spot = data.spot;
        break;
    case EmitterType::kDirectional:
        directional = data.directional;
        break;
    case EmitterType::kSun:
        sun = data.sun;
        break;
    case EmitterType::kEnvMap:
        envmap = data.envmap;
        break;
    case EmitterType::kConstant:
        constant = data.constant;
        break;
    }
}

QUALIFIER_D_H Emitter::Emitter() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Emitter::Emitter(const uint32_t id, const EmitterInfo &info,
                               Texture *texture_buffer)
    : id_(id)
{
    data_.type = info.type;
    switch (info.type)
    {
    case EmitterType::kPoint:
        data_.point = info.point;
        break;
    case EmitterType::kSpot:
        data_.spot.cutoff_angle = info.spot.cutoff_angle;
        data_.spot.cos_cutoff_angle = cosf(info.spot.cutoff_angle);
        data_.spot.uv_factor = tanf(info.spot.cutoff_angle);
        data_.spot.beam_width = info.spot.beam_width;
        data_.spot.cos_beam_width = cosf(info.spot.beam_width);
        data_.spot.transition_width_rcp =
            1.0f / (info.spot.cutoff_angle - info.spot.beam_width);
        data_.spot.intensity = info.spot.intensity;
        data_.spot.texture = (info.spot.id_texture == kInvalidId)
                                 ? nullptr
                                 : texture_buffer + info.spot.id_texture;
        data_.spot.position = TransformPoint(info.spot.to_world, {0, 0, 0});
        data_.spot.to_local = info.spot.to_world.Inverse();
        break;
    case EmitterType::kDirectional:
        data_.directional = info.directional;
        break;
    case EmitterType::kSun:
        data_.sun.cos_cutoff_angle = info.sun.cos_cutoff_angle;
        data_.sun.texture = texture_buffer + info.sun.id_texture;
        data_.sun.direction = info.sun.direction;
        data_.sun.radiance = info.sun.radiance;
        break;
    case EmitterType::kEnvMap:
        data_.envmap.radiance = texture_buffer + info.envmap.id_radiance;
        data_.envmap.to_world = info.envmap.to_world;
        data_.envmap.to_local = info.envmap.to_world.Inverse();
        break;
    case EmitterType::kConstant:
        data_.constant = info.constant;
    }
}

QUALIFIER_D_H void Emitter::InitEnvMap(const int width, const int height,
                                       const float normalization, float *pixels)
{
    data_.envmap.width = width;
    data_.envmap.height = height;
    data_.envmap.normalization = normalization;
    data_.envmap.cdf_cols = pixels;
    data_.envmap.cdf_rows = pixels + height + 1;
    data_.envmap.weight_rows = pixels + (height + 1) + height;
}

QUALIFIER_D_H EmitterSampleRec Emitter::Sample(const Vec3 &origin,
                                               const float xi_0,
                                               const float xi_1) const
{
    EmitterSampleRec rec;
    switch (data_.type)
    {
    case EmitterType::kPoint:
        SamplePointLight(data_.point, origin, xi_0, xi_1, &rec);
        break;
    case EmitterType::kSpot:
        SampleSpotLight(data_.spot, origin, xi_0, xi_1, &rec);
        break;
    case EmitterType::kDirectional:
        SampleDirectionalLight(data_.directional, origin, xi_0, xi_1, &rec);
        break;
    case EmitterType::kSun:
        SampleSun(data_.sun, origin, xi_0, xi_1, &rec);
        break;
    case EmitterType::kEnvMap:
        SampleEnvMap(data_.envmap, origin, xi_0, xi_1, &rec);
        break;
    case EmitterType::kConstant:
        SampleConstantLight(data_.constant, origin, xi_0, xi_1, &rec);
        break;
    }
    return rec;
}

QUALIFIER_D_H Vec3 Emitter::Evaluate(const EmitterSampleRec &rec) const
{
    switch (data_.type)
    {
    case EmitterType::kPoint:
        return EvaluatePointLight(data_.point, &rec);
        break;
    case EmitterType::kSpot:
        return EvaluateSpotLight(data_.spot, &rec);
        break;
    case EmitterType::kDirectional:
        return EvaluateDirectionalLight(data_.directional, &rec);
        break;
    case EmitterType::kSun:
        return EvaluateSun(data_.sun, &rec);
        break;
    case EmitterType::kEnvMap:
        return EvaluateEnvMap(data_.envmap, &rec);
        break;
    case EmitterType::kConstant:
        return EvaluateConstantLight(data_.constant, &rec);
        break;
    }
    return {};
}

QUALIFIER_D_H Vec3 Emitter::Evaluate(const Vec3 &look_dir) const
{
    switch (data_.type)
    {
    case EmitterType::kSun:
        return EvaluateSun(data_.sun, look_dir);
        break;
    case EmitterType::kEnvMap:
        return EvaluateEnvMap(data_.envmap, look_dir);
        break;
    case EmitterType::kConstant:
        return EvaluateConstantLight(data_.constant, look_dir);
        break;
    }
    return {};
}

QUALIFIER_D_H float Emitter::Pdf(const Vec3 &look_dir) const
{
    switch (data_.type)
    {
    case EmitterType::kEnvMap:
        return PdfEnvMap(data_.envmap, look_dir);
        break;
    case EmitterType::kConstant:
        return PdfConstantLight(data_.constant, look_dir);
        break;
    }
    return 0;
}

} // namespace csrt
