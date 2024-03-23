#include "csrt/renderer/bsdfs/bsdf.cuh"

#include <exception>

namespace csrt
{

QUALIFIER_D_H BsdfInfo::BsdfInfo()
    : type(BsdfType::kNone), twosided(false), id_opacity(kInvalidId),
      id_bump_map(kInvalidId), area_light{}, diffuse{}, rough_diffuse{},
      conductor{}, dielectric{}, plastic{}
{
}

// QUALIFIER_D_H BsdfInfo::Info(const BsdfInfo &info)
//     : type(info.type), twosided(info.twosided), id_opacity(info.id_opacity),
//       id_bump_map(info.id_bump_map)
// {
//     switch (info.type)
//     {
//     case BsdfType::kNone:
//         break;
//     case BsdfType::kAreaLight:
//         area_light = info.area_light;
//         break;
//     case BsdfType::kDiffuse:
//         diffuse = info.diffuse;
//         break;
//     case BsdfType::kRoughDiffuse:
//         rough_diffuse = info.rough_diffuse;
//         break;
//     case BsdfType::kConductor:
//         conductor = info.conductor;
//         break;
//     case BsdfType::kDielectric:
//     case BsdfType::kThinDielectric:
//         dielectric = info.dielectric;
//         break;
//     case BsdfType::kPlastic:
//         plastic = info.plastic;
//         break;
//     }
// }

// QUALIFIER_D_H void BsdfInfo::operator=(const BsdfInfo &info)
// {
//     type = info.type;
//     twosided = info.twosided;
//     id_opacity = info.id_opacity;
//     id_bump_map = info.id_bump_map;
//     switch (info.type)
//     {
//     case BsdfType::kNone:
//         break;
//     case BsdfType::kAreaLight:
//         area_light = info.area_light;
//         break;
//     case BsdfType::kDiffuse:
//         diffuse = info.diffuse;
//         break;
//     case BsdfType::kRoughDiffuse:
//         rough_diffuse = info.rough_diffuse;
//         break;
//     case BsdfType::kConductor:
//         conductor = info.conductor;
//         break;
//     case BsdfType::kDielectric:
//     case BsdfType::kThinDielectric:
//         dielectric = info.dielectric;
//         break;
//     case BsdfType::kPlastic:
//         plastic = info.plastic;
//         break;
//     }
// }

QUALIFIER_D_H BsdfData::BsdfData()
    : type(BsdfType::kNone), twosided(false), opacity(nullptr),
      bump_map(nullptr), area_light{}, diffuse{}, rough_diffuse{}, conductor{},
      dielectric{}, plastic{}
{
}

// QUALIFIER_D_H Bsdf::Data::Data(const Bsdf::Data &info)
//     : type(info.type), twosided(info.twosided), opacity(info.opacity),
//       bump_map(info.bump_map)
// {
//     switch (info.type)
//     {
//     case BsdfType::kAreaLight:
//         area_light = info.area_light;
//         break;
//     case BsdfType::kDiffuse:
//         diffuse = info.diffuse;
//         break;
//     case BsdfType::kRoughDiffuse:
//         rough_diffuse = info.rough_diffuse;
//         break;
//     case BsdfType::kConductor:
//         conductor = info.conductor;
//         break;
//     case BsdfType::kDielectric:
//     case BsdfType::kThinDielectric:
//         dielectric = info.dielectric;
//         break;
//     case BsdfType::kPlastic:
//         plastic = info.plastic;
//         break;
//     }
// }

QUALIFIER_D_H void BsdfData::operator=(const BsdfData &info)
{
    type = info.type;
    twosided = info.twosided;
    opacity = info.opacity;
    bump_map = info.bump_map;
    switch (info.type)
    {
    case BsdfType::kAreaLight:
        area_light = info.area_light;
        break;
    case BsdfType::kDiffuse:
        diffuse = info.diffuse;
        break;
    case BsdfType::kRoughDiffuse:
        rough_diffuse = info.rough_diffuse;
        break;
    case BsdfType::kConductor:
        conductor = info.conductor;
        break;
    case BsdfType::kDielectric:
    case BsdfType::kThinDielectric:
        dielectric = info.dielectric;
        break;
    case BsdfType::kPlastic:
        plastic = info.plastic;
        break;
    }
}

QUALIFIER_D_H Vec3 BsdfSampleRec::ToLocal(const Vec3 &v) const
{
    return Normalize({Dot(v, tangent), Dot(v, bitangent), Dot(v, normal)});
}

QUALIFIER_D_H Vec3 BsdfSampleRec::ToWorld(const Vec3 &v) const
{
    return Normalize(v.x * tangent + v.y * bitangent + v.z * normal);
}

QUALIFIER_D_H Bsdf::Bsdf() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Bsdf::Bsdf(const uint32_t id, const BsdfInfo &info,
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
    case BsdfType::kAreaLight:
        data_.area_light.radiance =
            texture_buffer + info.area_light.id_radiance;
        break;
    case BsdfType::kDiffuse:
        data_.diffuse.diffuse_reflectance =
            texture_buffer + info.diffuse.id_diffuse_reflectance;
        break;
    case BsdfType::kRoughDiffuse:
        data_.rough_diffuse.diffuse_reflectance =
            texture_buffer + info.rough_diffuse.id_diffuse_reflectance;
        data_.rough_diffuse.roughness =
            texture_buffer + info.rough_diffuse.id_roughness;
        break;
    case BsdfType::kConductor:
        data_.conductor.roughness_u =
            texture_buffer + info.conductor.id_roughness_u;
        data_.conductor.roughness_v =
            texture_buffer + info.conductor.id_roughness_v;
        data_.conductor.specular_reflectance =
            texture_buffer + info.conductor.id_specular_reflectance;
        data_.conductor.reflectivity = info.conductor.reflectivity;
        data_.conductor.edgetint = info.conductor.edgetint;
        break;
    case BsdfType::kThinDielectric:
    case BsdfType::kDielectric:
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
    case BsdfType::kPlastic:
        data_.plastic.roughness = texture_buffer + info.plastic.id_roughness;
        data_.plastic.diffuse_reflectance =
            texture_buffer + info.plastic.id_diffuse_reflectance;
        data_.plastic.specular_reflectance =
            texture_buffer + info.plastic.id_specular_reflectance;
        data_.plastic.reflectivity =
            (Sqr(info.plastic.eta - 1.0f) / Sqr(info.plastic.eta + 1.0f));
        data_.plastic.F_avg = AverageFresnel(info.plastic.eta);
        break;
    }
}

QUALIFIER_D_H void Bsdf::Evaluate(BsdfSampleRec *rec) const
{
    switch (data_.type)
    {
    case BsdfType::kDiffuse:
        EvaluateDiffuse(data_.diffuse, rec);
        break;
    case BsdfType::kRoughDiffuse:
        EvaluateRoughDiffuse(data_.rough_diffuse, rec);
        break;
    case BsdfType::kConductor:
        EvaluateConductor(data_.conductor, rec);
        break;
    case BsdfType::kDielectric:
        EvaluateDielectric(data_.dielectric, rec);
        break;
    case BsdfType::kThinDielectric:
        EvaluateThinDielectric(data_.dielectric, rec);
        break;
    case BsdfType::kPlastic:
        EvaluatePlastic(data_.plastic, rec);
        break;
    }
}

QUALIFIER_D_H void Bsdf::Sample(uint32_t *seed, BsdfSampleRec *rec) const
{
    switch (data_.type)
    {
    case BsdfType::kDiffuse:
        SampleDiffuse(data_.diffuse, seed, rec);
        break;
    case BsdfType::kRoughDiffuse:
        SampleRoughDiffuse(data_.rough_diffuse, seed, rec);
        break;
    case BsdfType::kConductor:
        SampleConductor(data_.conductor, seed, rec);
        break;
    case BsdfType::kDielectric:
        SampleDielectric(data_.dielectric, seed, rec);
        break;
    case BsdfType::kThinDielectric:
        SampleThinDielectric(data_.dielectric, seed, rec);
        break;
    case BsdfType::kPlastic:
        SamplePlastic(data_.plastic, seed, rec);
        break;
    }
}

QUALIFIER_D_H Vec3 Bsdf::GetRadiance(const Vec2 &texcoord) const
{
    if (data_.type == BsdfType::kAreaLight)
    {
        return data_.area_light.radiance->GetColor(texcoord);
    }
    else
    {
        return {};
    }
}

QUALIFIER_D_H bool Bsdf::IsEmitter() const
{
    return data_.type == BsdfType::kAreaLight;
}

QUALIFIER_D_H bool Bsdf::IsTransparent(const Vec2 &texcoord,
                                       uint32_t *seed) const
{
    return data_.opacity && data_.opacity->IsTransparent(texcoord, seed);
}

} // namespace csrt