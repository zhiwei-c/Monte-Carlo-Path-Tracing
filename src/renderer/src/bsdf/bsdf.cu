#include "bsdf.cuh"

#include <exception>

namespace csrt
{

QUALIFIER_D_H BSDF::Data::Data()
    : type(BSDF::Type::kNone), twosided(false), opacity(nullptr),
      bump_map(nullptr), area_light{}, diffuse{}, rough_diffuse{}, conductor{},
      dielectric{}, plastic{}
{
}

QUALIFIER_D_H BSDF::Data::Data(const BSDF::Data &info)
    : type(info.type), twosided(info.twosided), opacity(info.opacity),
      bump_map(info.bump_map)
{
    switch (info.type)
    {
    case BSDF::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case BSDF::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case BSDF::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case BSDF::Type::kConductor:
        conductor = info.conductor;
        break;
    case BSDF::Type::kDielectric:
    case BSDF::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case BSDF::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H void BSDF::Data::operator=(const BSDF::Data &info)
{
    type = info.type;
    twosided = info.twosided;
    opacity = info.opacity;
    bump_map = info.bump_map;
    switch (info.type)
    {
    case BSDF::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case BSDF::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case BSDF::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case BSDF::Type::kConductor:
        conductor = info.conductor;
        break;
    case BSDF::Type::kDielectric:
    case BSDF::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case BSDF::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H BSDF::Info::Info()
    : type(BSDF::Type::kNone), twosided(false), id_opacity(kInvalidId),
      id_bump_map(kInvalidId), area_light{}, diffuse{}, rough_diffuse{},
      conductor{}, dielectric{}, plastic{}
{
}

QUALIFIER_D_H BSDF::Info::Info(const BSDF::Info &info)
    : type(info.type), twosided(info.twosided), id_opacity(info.id_opacity),
      id_bump_map(info.id_bump_map)
{
    switch (info.type)
    {
    case BSDF::Type::kNone:
        break;
    case BSDF::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case BSDF::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case BSDF::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case BSDF::Type::kConductor:
        conductor = info.conductor;
        break;
    case BSDF::Type::kDielectric:
    case BSDF::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case BSDF::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H void BSDF::Info::operator=(const BSDF::Info &info)
{
    type = info.type;
    twosided = info.twosided;
    id_opacity = info.id_opacity;
    id_bump_map = info.id_bump_map;
    switch (info.type)
    {
    case BSDF::Type::kNone:
        break;
    case BSDF::Type::kAreaLight:
        area_light = info.area_light;
        break;
    case BSDF::Type::kDiffuse:
        diffuse = info.diffuse;
        break;
    case BSDF::Type::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case BSDF::Type::kConductor:
        conductor = info.conductor;
        break;
    case BSDF::Type::kDielectric:
    case BSDF::Type::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case BSDF::Type::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H Vec3 BSDF::SampleRec::ToLocal(const Vec3 &v) const
{
    return Normalize({Dot(v, tangent), Dot(v, bitangent), Dot(v, normal)});
}

QUALIFIER_D_H Vec3 BSDF::SampleRec::ToWorld(const Vec3 &v) const
{
    return Normalize(v.x * tangent + v.y * bitangent + v.z * normal);
}

QUALIFIER_D_H BSDF::BSDF() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H BSDF::BSDF(const uint32_t id, const BSDF::Info &info,
                         Texture *texture_buffer)
    : id_(id)
{
    data_.type = info.type;
    data_.twosided = info.twosided;
    data_.opacity = info.id_opacity == kInvalidId
                        ? nullptr
                        : texture_buffer + info.id_opacity;
    data_.bump_map = info.id_bump_map == kInvalidId
                         ? nullptr
                         : texture_buffer + info.id_bump_map;
    switch (info.type)
    {
    case BSDF::Type::kAreaLight:
        data_.area_light.radiance =
            texture_buffer + info.area_light.id_radiance;
        break;
    case BSDF::Type::kDiffuse:
        data_.diffuse.diffuse_reflectance =
            texture_buffer + info.diffuse.id_diffuse_reflectance;
        break;
    case BSDF::Type::kRoughDiffuse:
        data_.rough_diffuse.diffuse_reflectance =
            texture_buffer + info.rough_diffuse.id_diffuse_reflectance;
        data_.rough_diffuse.roughness =
            texture_buffer + info.rough_diffuse.id_roughness;
        break;
    case BSDF::Type::kConductor:
        data_.conductor.roughness_u =
            texture_buffer + info.conductor.id_roughness_u;
        data_.conductor.roughness_v =
            texture_buffer + info.conductor.id_roughness_v;
        data_.conductor.specular_reflectance =
            texture_buffer + info.conductor.id_specular_reflectance;
        data_.conductor.reflectivity = info.conductor.reflectivity;
        data_.conductor.edgetint = info.conductor.edgetint;
        break;
    case BSDF::Type::kThinDielectric:
    case BSDF::Type::kDielectric:
        data_.twosided = true;
        data_.dielectric.roughness_u =
            texture_buffer + info.dielectric.id_roughness_u;
        data_.dielectric.roughness_v =
            texture_buffer + info.dielectric.id_roughness_v;
        data_.dielectric.specular_reflectance =
            texture_buffer + info.dielectric.id_specular_reflectance;
        data_.dielectric.specular_transmittance =
            texture_buffer + info.dielectric.id_specular_transmittance;
        data_.dielectric.eta = info.dielectric.eta;
        data_.dielectric.eta_inv = 1.0f / info.dielectric.eta;
        data_.dielectric.reflectivity =
            (Sqr(info.dielectric.eta - 1.0f) / Sqr(info.dielectric.eta + 1.0f));
        break;
    case BSDF::Type::kPlastic:
        data_.plastic.roughness = texture_buffer + info.plastic.id_roughness;
        data_.plastic.diffuse_reflectance =
            texture_buffer + info.plastic.id_diffuse_reflectance;
        data_.plastic.specular_reflectance =
            texture_buffer + info.plastic.id_specular_reflectance;
        data_.plastic.reflectivity =
            (Sqr(info.plastic.eta - 1.0f) / Sqr(info.plastic.eta + 1.0f));
        data_.plastic.F_avg = BSDF::AverageFresnel(info.plastic.eta);
        break;
    }
}

QUALIFIER_D_H void BSDF::Evaluate(BSDF::SampleRec *rec) const
{
    switch (data_.type)
    {
    case BSDF::Type::kDiffuse:
        EvaluateDiffuse(rec);
        break;
    case BSDF::Type::kRoughDiffuse:
        EvaluateRoughDiffuse(rec);
        break;
    case BSDF::Type::kConductor:
        EvaluateConductor(rec);
        break;
    case BSDF::Type::kDielectric:
        EvaluateDielectric(rec);
        break;
    case BSDF::Type::kThinDielectric:
        EvaluateThinDielectric(rec);
        break;
    case BSDF::Type::kPlastic:
        EvaluatePlastic(rec);
        break;
    }
}

QUALIFIER_D_H void BSDF::Sample(uint32_t *seed, BSDF::SampleRec *rec) const
{
    switch (data_.type)
    {
    case BSDF::Type::kDiffuse:
        SampleDiffuse(seed, rec);
        break;
    case BSDF::Type::kRoughDiffuse:
        SampleRoughDiffuse(seed, rec);
        break;
    case BSDF::Type::kConductor:
        SampleConductor(seed, rec);
        break;
    case BSDF::Type::kDielectric:
        SampleDielectric(seed, rec);
        break;
    case BSDF::Type::kThinDielectric:
        SampleThinDielectric(seed, rec);
        break;
    case BSDF::Type::kPlastic:
        SamplePlastic(seed, rec);
        break;
    }
}

QUALIFIER_D_H Vec3 BSDF::GetRadiance(const Vec2 &texcoord) const
{
    if (data_.type == BSDF::Type::kAreaLight)
    {
        return data_.area_light.radiance->GetColor(texcoord);
    }
    else
    {
        return {};
    }
}

QUALIFIER_D_H bool BSDF::IsEmitter() const
{
    return data_.type == BSDF::Type::kAreaLight;
}

QUALIFIER_D_H bool BSDF::IsTransparent(const Vec2 &texcoord,
                                       uint32_t *seed) const
{
    return data_.opacity && data_.opacity->IsTransparent(texcoord, seed);
}

QUALIFIER_D_H float BSDF::AverageFresnel(const float eta)
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

} // namespace csrt