#ifndef CSRT__RENDERER__BSDF__BSDF_HPP
#define CSRT__RENDERER__BSDF__BSDF_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../textures/texture.hpp"
#include "conductor.hpp"
#include "dielectric.hpp"
#include "diffuse.hpp"
#include "plastic.hpp"
#include "rough_diffuse.hpp"
#include "thin_dielectric.hpp"

namespace csrt
{

enum class BsdfType
{
    kNone,
    kAreaLight,
    kDiffuse,
    kRoughDiffuse,
    kConductor,
    kDielectric,
    kThinDielectric,
    kPlastic,
};

struct AreaLightInfo
{
    float weight = 1;
    uint32_t id_radiance = kInvalidId;
};

struct AreaLightData
{
    Texture *radiance = nullptr;
};

struct BsdfInfo
{
    BsdfType type;
    bool twosided;
    uint32_t id_opacity;
    uint32_t id_bump_map;
    union
    {
        AreaLightInfo area_light;
        DiffuseInfo diffuse;
        RoughDiffuseInfo rough_diffuse;
        ConductorInfo conductor;
        DielectricInfo dielectric;
        PlasticInfo plastic;
    };

    QUALIFIER_D_H BsdfInfo();
    QUALIFIER_D_H ~BsdfInfo() {}
};

struct BsdfData
{
    BsdfType type;
    bool twosided;
    Texture *opacity;
    Texture *bump_map;
    union
    {
        AreaLightData area_light;
        DiffuseData diffuse;
        RoughDiffuseData rough_diffuse;
        ConductorData conductor;
        DielectricData dielectric;
        PlasticData plastic;
    };

    QUALIFIER_D_H BsdfData();
    QUALIFIER_D_H ~BsdfData() {}
    QUALIFIER_D_H void operator=(const BsdfData &data);
};

struct BsdfSampleRec
{
    bool valid = false;
    bool inside = false;
    float pdf = 0;
    Vec2 texcoord = {};
    Vec3 wi = {};
    Vec3 wo = {};
    Vec3 position = {};
    Vec3 normal = {};
    Vec3 tangent = {};
    Vec3 bitangent = {};
    Vec3 attenuation = {};

    QUALIFIER_D_H Vec3 ToLocal(const Vec3 &v) const;
    QUALIFIER_D_H Vec3 ToWorld(const Vec3 &v) const;
};

class Bsdf
{
public:
    QUALIFIER_D_H Bsdf();
    QUALIFIER_D_H Bsdf(const uint32_t id, const BsdfInfo &info,
                       Texture *texture_buffer, float *brdf_avg_buffer,
                       float *albedo_avg_buffer);

    QUALIFIER_D_H void Evaluate(BsdfSampleRec *rec) const;
    QUALIFIER_D_H void Sample(uint32_t *seed, BsdfSampleRec *rec) const;
    QUALIFIER_D_H Vec3 GetRadiance(const Vec2 &texcoord) const;
    QUALIFIER_D_H bool IsEmitter() const;
    QUALIFIER_D_H bool IsTwosided() const { return data_.twosided; }
    QUALIFIER_D_H bool IsTransparent(const Vec2 &texcoord,
                                     uint32_t *seed) const;

private:
    uint32_t id_;
    BsdfData data_;
};

} // namespace csrt

#endif