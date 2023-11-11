#include "bsdf.cuh"

#include <exception>

namespace rt
{

QUALIFIER_D_H Bsdf::Data::Data()
    : type(Bsdf::Type::kNone), twosided(false), id_opacity(kInvalidId),
      id_bump_map(kInvalidId), texture_buffer(nullptr), area_light{}, diffuse{},
      rough_diffuse{}, conductor{}, dielectric{}, plastic{}
{
}

QUALIFIER_D_H Bsdf::Data::Data(const Bsdf::Data &info)
    : type(info.type), twosided(info.twosided), id_opacity(info.id_opacity),
      id_bump_map(info.id_bump_map), texture_buffer(info.texture_buffer)
{
    switch (info.type)
    {
    case Bsdf::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case Bsdf::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case Bsdf::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case Bsdf::Type::kConductor:
        conductor = info.conductor;
        break;
    case Bsdf::Type::kDielectric:
    case Bsdf::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case Bsdf::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H void Bsdf::Data::operator=(const Bsdf::Data &info)
{
    type = info.type;
    twosided = info.twosided;
    id_opacity = info.id_opacity;
    id_bump_map = info.id_bump_map;
    texture_buffer = info.texture_buffer;
    switch (info.type)
    {
    case Bsdf::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case Bsdf::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case Bsdf::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case Bsdf::Type::kConductor:
        conductor = info.conductor;
        break;
    case Bsdf::Type::kDielectric:
    case Bsdf::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case Bsdf::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

Bsdf::Info::Info()
    : type(Bsdf::Type::kNone), twosided(false), id_opacity(kInvalidId),
      id_bump_map(kInvalidId), area_light{}, diffuse{}, rough_diffuse{},
      conductor{}, dielectric{}, plastic{}
{
}

Bsdf::Info::Info(const Bsdf::Info &info)
    : type(info.type), twosided(info.twosided), id_opacity(info.id_opacity),
      id_bump_map(info.id_bump_map)
{
    switch (info.type)
    {
    case Bsdf::Type::kNone:
        break;
    case Bsdf::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case Bsdf::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case Bsdf::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case Bsdf::Type::kConductor:
        conductor = info.conductor;
        break;
    case Bsdf::Type::kDielectric:
    case Bsdf::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case Bsdf::Type::kPlastic:
        plastic = info.plastic;
        break;
    default:
        throw std::exception("unknow instance type.");
        break;
    }
}

void Bsdf::Info::operator=(const Bsdf::Info &info)
{
    type = info.type;
    twosided = info.twosided;
    id_opacity = info.id_opacity;
    id_bump_map = info.id_bump_map;
    switch (info.type)
    {
    case Bsdf::Type::kNone:
        break;
    case Bsdf::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case Bsdf::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case Bsdf::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case Bsdf::Type::kConductor:
        conductor = info.conductor;
        break;
    case Bsdf::Type::kDielectric:
    case Bsdf::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case Bsdf::Type::kPlastic:
        plastic = info.plastic;
        break;
    default:
        throw std::exception("unknow instance type.");
        break;
    }
}

Bsdf::Info Bsdf::Info::CreateAreaLight(const uint32_t id_radiance,
                                       const float weight, const bool twosided,
                                       const uint32_t id_opacity,
                                       const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kAreaLight;
    info.area_light.id_radiance = id_radiance;
    info.area_light.weight = weight;
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

Bsdf::Info Bsdf::Info::CreateDiffuse(const uint32_t id_diffuse_reflectance,
                                     const bool twosided,
                                     const uint32_t id_opacity,
                                     const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kDiffuse;
    info.diffuse.id_diffuse_reflectance = id_diffuse_reflectance;
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

Bsdf::Info Bsdf::Info::CreateRoughDiffuse(const bool use_fast_approx,
                                          const uint32_t id_diffuse_reflectance,
                                          const uint32_t id_roughness,
                                          const bool twosided,
                                          const uint32_t id_opacity,
                                          const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kRoughDiffuse;
    info.rough_diffuse.use_fast_approx = use_fast_approx;
    info.rough_diffuse.id_diffuse_reflectance = id_diffuse_reflectance;
    info.rough_diffuse.id_roughness = id_roughness;
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

Bsdf::Info Bsdf::Info::CreateConductor(
    const uint32_t id_roughness_u, const uint32_t id_roughness_v,
    const uint32_t id_specular_reflectance, const Vec3 &eta, const Vec3 &k,
    const bool twosided, const uint32_t id_opacity, const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kConductor;
    info.conductor.id_roughness_u = id_roughness_u;
    info.conductor.id_roughness_v = id_roughness_v;
    info.conductor.id_specular_reflectance = id_specular_reflectance;
    info.conductor.reflectivity =
        (Sqr(eta - 1.0f) + Sqr(k)) / (Sqr(eta + 1.0f) + Sqr(k));
    const Vec3 temp1 = 1.0f + Sqrt(info.conductor.reflectivity),
               temp2 = 1.0f - Sqrt(info.conductor.reflectivity),
               temp3 = ((1.0f - info.conductor.reflectivity) /
                        (1.0 + info.conductor.reflectivity));
    info.conductor.edgetint = (temp1 - eta * temp2) / (temp1 - temp3 * temp2);
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

Bsdf::Info Bsdf::Info::CreateDielectric(
    bool is_thin, const uint32_t id_roughness_u, const uint32_t id_roughness_v,
    const uint32_t id_specular_reflectance,
    const uint32_t id_specular_transmittance, const float eta,
    const bool twosided, const uint32_t id_opacity, const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = is_thin ? Bsdf::Type::kThinDielectric : Bsdf::Type::kDielectric;
    info.dielectric.id_roughness_u = id_roughness_u;
    info.dielectric.id_roughness_v = id_roughness_v;
    info.dielectric.id_specular_reflectance = id_specular_reflectance;
    info.dielectric.id_specular_transmittance = id_specular_transmittance;
    info.dielectric.eta = eta;
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

Bsdf::Info Bsdf::Info::CreatePlastic(const float eta,
                                     const uint32_t id_roughness,
                                     const uint32_t id_diffuse_reflectance,
                                     const uint32_t id_specular_reflectance,
                                     const bool twosided,
                                     const uint32_t id_opacity,
                                     const uint32_t id_bump_map)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kPlastic;
    info.plastic.eta = eta;
    info.plastic.id_roughness = id_roughness;
    info.plastic.id_diffuse_reflectance = id_diffuse_reflectance;
    info.plastic.id_specular_reflectance = id_specular_reflectance;
    info.twosided = twosided;
    info.id_opacity = id_opacity;
    info.id_bump_map = id_bump_map;
    return info;
}

QUALIFIER_D_H Bsdf::Bsdf() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Bsdf::Bsdf(const uint32_t id, const Bsdf::Data &data)
    : id_(id), data_(data)
{
}

QUALIFIER_D_H void Bsdf::Evaluate(SamplingRecord *rec) const
{
    switch (data_.type)
    {
    case Bsdf::Type::kDiffuse:
        EvaluateDiffuse(rec);
        break;
    case Bsdf::Type::kRoughDiffuse:
        EvaluateRoughDiffuse(rec);
        break;
    case Bsdf::Type::kConductor:
        EvaluateConductor(rec);
        break;
    case Bsdf::Type::kDielectric:
        EvaluateDielectric(rec);
        break;
    case Bsdf::Type::kThinDielectric:
        EvaluateThinDielectric(rec);
        break;
    case Bsdf::Type::kPlastic:
        EvaluatePlastic(rec);
        break;
    }
}

QUALIFIER_D_H void Bsdf::Sample(const Vec3 &xi, SamplingRecord *rec) const
{
    switch (data_.type)
    {
    case Bsdf::Type::kDiffuse:
        SampleDiffuse(xi, rec);
        break;
    case Bsdf::Type::kRoughDiffuse:
        SampleRoughDiffuse(xi, rec);
        break;
    case Bsdf::Type::kConductor:
        SampleConductor(xi, rec);
        break;
    case Bsdf::Type::kDielectric:
        SampleDielectric(xi, rec);
        break;
    case Bsdf::Type::kThinDielectric:
        SampleThinDielectric(xi, rec);
        break;
    case Bsdf::Type::kPlastic:
        SamplePlastic(xi, rec);
        break;
    }
}

QUALIFIER_D_H Vec3 Bsdf::GetRadiance(const Vec2 &texcoord) const
{
    if (data_.type == Bsdf::Type::kAreaLight)
    {
        const Texture &radiance =
            data_.texture_buffer[data_.area_light.id_radiance];
        return radiance.GetColor(texcoord);
    }
    else
    {
        return {};
    }
}

QUALIFIER_D_H bool Bsdf::IsEmitter() const
{
    return data_.type == Bsdf::Type::kAreaLight;
}

QUALIFIER_D_H bool Bsdf::IsTransparent(const Vec2 &texcoord,
                                       const float xi) const
{
    if (data_.id_opacity == kInvalidId)
        return false;
    const Texture &opacity = data_.texture_buffer[data_.id_opacity];
    return opacity.IsTransparent(texcoord, xi);
}

float Bsdf::AverageFresnel(const float eta)
{
    if (eta < 1.0)
    {
        /* Fit by Egan and Hilgeman (1973). Works reasonably well for
            "normal" IOR values (<2).
            Max rel. error in 1.0 - 1.5 : 0.1%
            Max rel. error in 1.5 - 2   : 0.6%
            Max rel. error in 2.0 - 5   : 9.5%
        */
        return -1.4399f * Sqr(eta) + 0.7099f * eta + 0.6681f + 0.0636f / eta;
    }
    else
    {
        /* Fit by d'Eon and Irving (2011)

            Maintains a good accuracy even for unrealistic IOR values.

            Max rel. error in 1.0 - 2.0   : 0.1%
            Max rel. error in 2.0 - 10.0  : 0.2%
        */
        float inv_eta = 1.0f / eta, inv_eta_2 = inv_eta * inv_eta,
              inv_eta_3 = inv_eta_2 * inv_eta, inv_eta_4 = inv_eta_3 * inv_eta,
              inv_eta_5 = inv_eta_4 * inv_eta;
        return 0.919317f - 3.4793f * inv_eta + 6.75335f * inv_eta_2 -
               7.80989f * inv_eta_3 + 4.98554f * inv_eta_4 -
               1.36881f * inv_eta_5;
    }
}

} // namespace rt