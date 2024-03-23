#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"
#include "../textures/texture.cuh"
#include "conductor.cuh"
#include "dielectric.cuh"
#include "diffuse.cuh"
#include "plastic.cuh"
#include "rough_diffuse.cuh"
#include "thin_dielectric.cuh"

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
    // QUALIFIER_D_H Info(const Info &info);
    // QUALIFIER_D_H void operator=(const Info &info);
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
    // QUALIFIER_D_H Data(const BSDF::Data &info);
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
                       Texture *texture_buffer);

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
