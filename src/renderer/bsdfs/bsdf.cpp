#include "csrt/renderer/bsdfs/bsdf.hpp"

#include <exception>

namespace
{

using namespace csrt;

QUALIFIER_D_H float AverageFresnel(const float eta)
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

QUALIFIER_D_H Vec3 AverageFresnel(const Vec3 &reflectivity,
                                  const Vec3 &edgetint)
{
    return Vec3(0.087237f) + 0.0230685f * edgetint -
           0.0864902f * edgetint * edgetint +
           0.0774594f * edgetint * edgetint * edgetint +
           0.782654f * reflectivity - 0.136432f * reflectivity * reflectivity +
           0.278708f * reflectivity * reflectivity * reflectivity +
           0.19744f * edgetint * reflectivity +
           0.0360605f * edgetint * edgetint * reflectivity -
           0.2586f * edgetint * reflectivity * reflectivity;
}

} // namespace

namespace csrt
{

QUALIFIER_D_H BsdfInfo::BsdfInfo()
    : type(BsdfType::kNone), twosided(false), id_opacity(kInvalidId),
      id_bump_map(kInvalidId), area_light{}
{
}

QUALIFIER_D_H BsdfData::BsdfData()
    : type(BsdfType::kNone), twosided(false), opacity(nullptr),
      bump_map(nullptr), area_light{}
{
}

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
                         Texture *texture_buffer, float *brdf_avg_buffer,
                         float *albedo_avg_buffer)
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
        data_.conductor.F_avg = AverageFresnel(info.conductor.reflectivity,
                                               info.conductor.edgetint);
        data_.conductor.brdf_avg_buffer = brdf_avg_buffer;
        data_.conductor.albedo_avg_buffer = albedo_avg_buffer;
        break;
    case BsdfType::kDielectric:
        data_.dielectric.F_avg = AverageFresnel(info.dielectric.eta);
        data_.dielectric.F_avg_inv = AverageFresnel(1.0f / info.dielectric.eta);
        data_.dielectric.brdf_avg_buffer = brdf_avg_buffer;
        data_.dielectric.albedo_avg_buffer = albedo_avg_buffer;
    case BsdfType::kThinDielectric:
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