#include "csrt/renderer/emitter.cuh"

namespace csrt
{

QUALIFIER_D_H Emitter::Data::Data()
    : type(Emitter::Type::kNone), point{}, spot{}, directional{}, sun{},
      envmap{}, constant{}
{
}

QUALIFIER_D_H Emitter::Data::Data(const Emitter::Data &info) : type(info.type)
{
    switch (info.type)
    {
    case Emitter::Type::kPoint:
        point = info.point;
        break;
    case Emitter::Type::kSpot:
        spot = info.spot;
        break;
    case Emitter::Type::kDirectional:
        directional = info.directional;
        break;
    case Emitter::Type::kSun:
        sun = info.sun;
        break;
    case Emitter::Type::kEnvMap:
        envmap = info.envmap;
        break;
    case Emitter::Type::kConstant:
        constant = info.constant;
        break;
    }
}

QUALIFIER_D_H void Emitter::Data::operator=(const Emitter::Data &info)
{
    type = info.type;
    switch (info.type)
    {
    case Emitter::Type::kPoint:
        point = info.point;
        break;
    case Emitter::Type::kSpot:
        spot = info.spot;
        break;
    case Emitter::Type::kDirectional:
        directional = info.directional;
        break;
    case Emitter::Type::kSun:
        sun = info.sun;
        break;
    case Emitter::Type::kEnvMap:
        envmap = info.envmap;
        break;
    case Emitter::Type::kConstant:
        constant = info.constant;
        break;
    }
}

QUALIFIER_D_H Emitter::Info::Info()
    : type(Emitter::Type::kNone), point{}, spot{}, directional{}, sun{},
      envmap{}, constant{}
{
}

QUALIFIER_D_H Emitter::Info::Info(const Emitter::Info &info) : type(info.type)
{
    switch (info.type)
    {
    case Emitter::Type::kPoint:
        point = info.point;
        break;
    case Emitter::Type::kSpot:
        spot = info.spot;
        break;
    case Emitter::Type::kDirectional:
        directional = info.directional;
        break;
    case Emitter::Type::kSun:
        sun = info.sun;
        break;
    case Emitter::Type::kEnvMap:
        envmap = info.envmap;
        break;
    case Emitter::Type::kConstant:
        constant = info.constant;
        break;
    }
}

QUALIFIER_D_H void Emitter::Info::operator=(const Emitter::Info &info)
{
    type = info.type;
    switch (info.type)
    {
    case Emitter::Type::kPoint:
        point = info.point;
        break;
    case Emitter::Type::kSpot:
        spot = info.spot;
        break;
    case Emitter::Type::kDirectional:
        directional = info.directional;
        break;
    case Emitter::Type::kSun:
        sun = info.sun;
        break;
    case Emitter::Type::kEnvMap:
        envmap = info.envmap;
        break;
    case Emitter::Type::kConstant:
        constant = info.constant;
        break;
    }
}

QUALIFIER_D_H Emitter::Emitter() : id_(kInvalidId), tlas_(nullptr), data_{} {}

QUALIFIER_D_H Emitter::Emitter(const uint32_t id, const Emitter::Info &info,
                               TLAS *tlas, Texture *texture_buffer)
    : id_(id), tlas_(tlas)
{
    data_.type = info.type;
    switch (info.type)
    {
    case Emitter::Type::kPoint:
        data_.point = info.point;
        break;
    case Emitter::Type::kSpot:
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
    case Emitter::Type::kDirectional:
        data_.directional = info.directional;
        break;
    case Emitter::Type::kSun:
        data_.sun.cos_cutoff_angle = info.sun.cos_cutoff_angle;
        data_.sun.texture = texture_buffer + info.sun.id_texture;
        data_.sun.direction = info.sun.direction;
        data_.sun.radiance = info.sun.radiance;
        break;
    case Emitter::Type::kEnvMap:
        data_.envmap.radiance = texture_buffer + info.envmap.id_radiance;
        data_.envmap.to_world = info.envmap.to_world;
        data_.envmap.to_local = info.envmap.to_world.Inverse();
        break;
    case Emitter::Type::kConstant:
        data_.constant = info.constant;
    }
}

QUALIFIER_D_H Emitter::SampleRec
Emitter::Sample(const Vec3 &origin, const float xi_0, const float xi_1) const
{
    switch (data_.type)
    {
    case Emitter::Type::kPoint:
        return SamplePoint(origin, xi_0, xi_1);
        break;
    case Emitter::Type::kSpot:
        return SampleSpot(origin, xi_0, xi_1);
        break;
    case Emitter::Type::kDirectional:
        return {true, true, kMaxFloat, data_.directional.direction};
        break;
    case Emitter::Type::kSun:
        return SampleSun(origin, xi_0, xi_1);
        break;
    case Emitter::Type::kEnvMap:
        return SampleEnvMap(origin, xi_0, xi_1);
        break;
    case Emitter::Type::kConstant:
        return {true, false, kMaxFloat, SampleSphereUniform(xi_0, xi_1)};
        break;
    }
    return {};
}

QUALIFIER_D_H Vec3 csrt::Emitter::Evaluate(const SampleRec &rec) const
{
    switch (data_.type)
    {
    case Emitter::Type::kPoint:
        return data_.point.intensity;
        break;
    case Emitter::Type::kSpot:
        return EvaluateSpot(rec);
        break;
    case Emitter::Type::kDirectional:
        return data_.directional.radiance;
        break;
    case Emitter::Type::kSun:
        return data_.sun.radiance;
        break;
    case Emitter::Type::kEnvMap:
        return EvaluateEnvMap(rec);
        break;
    case Emitter::Type::kConstant:
        return data_.constant.radiance;
        break;
    }
    return {};
}

QUALIFIER_D_H float Emitter::Pdf(const Vec3 &look_dir) const
{
    switch (data_.type)
    {
    case Emitter::Type::kEnvMap:
        return PdfEnvMap(look_dir);
        break;
    case Emitter::Type::kConstant:
        return k1Div4Pi;
        break;
    }
    return 0;
}

QUALIFIER_D_H Vec3 Emitter::Evaluate(const Vec3 &look_dir) const
{
    switch (data_.type)
    {
    case Emitter::Type::kSun:
        return EvaluateSun(look_dir);
        break;
    case Emitter::Type::kEnvMap:
        return EvaluateEnvMap(look_dir);
        break;
    case Emitter::Type::kConstant:
        return data_.constant.radiance;
        break;
    }
    return {};
}

QUALIFIER_D_H Emitter::SampleRec Emitter::SamplePoint(const Vec3 &origin,
                                                      const float xi_0,
                                                      const float xi_1) const
{
    const Vec3 vec = origin - data_.point.position;
    return {true, true, Length(vec), Normalize(vec)};
}

QUALIFIER_D_H Emitter::SampleRec Emitter::SampleSpot(const Vec3 &origin,
                                                     const float xi_0,
                                                     const float xi_1) const
{
    const Vec3 vec = origin - data_.spot.position;
    const Vec3 wi = Normalize(vec),
               dir_local = TransformVector(data_.spot.to_local, wi);
    if (dir_local.z < data_.spot.cos_cutoff_angle)
        return {};
    else
        return {true, true, Length(vec), wi};
}

QUALIFIER_D_H Emitter::SampleRec
Emitter::SampleSun(const Vec3 &origin, const float xi_0, const float xi_1) const
{
    const Vec3 dir_local =
        SampleConeUniform(data_.sun.cos_cutoff_angle, xi_0, xi_1);
    return {true, true, kMaxFloat,
            LocalToWorld(dir_local, data_.sun.direction)};
}

QUALIFIER_D_H Vec3 Emitter::EvaluateSpot(const SampleRec &rec) const
{
    const Vec3 dir = TransformVector(data_.spot.to_local, rec.wi);

    Vec3 fall_off = {1.0f, 1.0f, 1.0f};
    if (data_.spot.texture != nullptr)
    {
        const Vec2 texcoord = {
            0.5f + 0.5f * dir.x / (dir.z * data_.spot.uv_factor),
            0.5f + 0.5f * dir.y / (dir.z * data_.spot.uv_factor)};
        fall_off *= data_.spot.texture->GetColor(texcoord);
    }
    if (dir.z < data_.spot.cos_beam_width)
    {
        fall_off *= (data_.spot.cutoff_angle - acosf(dir.z)) *
                    data_.spot.transition_width_rcp;
    }
    return data_.spot.intensity * fall_off * Sqr(1.0f / rec.distance);
}

QUALIFIER_D_H Vec3 Emitter::EvaluateSun(const Vec3 &look_dir) const
{
    float phi = 0, theta = 0;
    CartesianToSpherical(look_dir, &theta, &phi, nullptr);
    const Vec2 texcoord = {phi * k1Div2Pi, theta * k1DivPi};
    return data_.sun.texture->GetColor(texcoord);
}

} // namespace csrt
