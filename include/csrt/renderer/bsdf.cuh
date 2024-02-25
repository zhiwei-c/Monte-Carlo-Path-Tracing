#pragma once

#include "../tensor.cuh"
#include "../utils.cuh"
#include "texture.cuh"

namespace csrt
{

class BSDF
{
public:
    enum Type
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

    struct Data
    {
        struct AreaLight
        {
            Texture *radiance = nullptr;
        };

        struct Diffuse
        {
            Texture *diffuse_reflectance = nullptr;
        };

        struct RoughDiffuse
        {
            bool use_fast_approx = true;
            Texture *diffuse_reflectance = nullptr;
            Texture *roughness = nullptr;
        };

        struct Conductor
        {
            Texture *roughness_u = nullptr;
            Texture *roughness_v = nullptr;
            Texture *specular_reflectance = nullptr;
            Vec3 reflectivity = {};
            Vec3 edgetint = {};
        };

        struct Dielectric
        {
            float reflectivity = 1.0f;
            float eta = 1.0f;
            float eta_inv = 1.0f;
            Texture *roughness_u = nullptr;
            Texture *roughness_v = nullptr;
            Texture *specular_reflectance = nullptr;
            Texture *specular_transmittance = nullptr;
        };

        struct Plastic
        {
            float reflectivity = 1.0f;
            float F_avg = 1.0f;
            Texture *roughness = nullptr;
            Texture *diffuse_reflectance = nullptr;
            Texture *specular_reflectance = nullptr;
        };

        BSDF::Type type;
        bool twosided;
        Texture *opacity;
        Texture *bump_map;
        union
        {
            AreaLight area_light;
            Diffuse diffuse;
            RoughDiffuse rough_diffuse;
            Conductor conductor;
            Dielectric dielectric;
            Plastic plastic;
        };

        QUALIFIER_D_H Data();
        QUALIFIER_D_H ~Data() {}
        QUALIFIER_D_H Data(const BSDF::Data &info);
        QUALIFIER_D_H void operator=(const BSDF::Data &info);
    };

    struct Info
    {
        struct AreaLight
        {
            float weight = 1;
            uint32_t id_radiance = kInvalidId;
        };

        struct Diffuse
        {
            uint32_t id_diffuse_reflectance = kInvalidId;
        };

        struct RoughDiffuse
        {
            bool use_fast_approx = true;
            uint32_t id_diffuse_reflectance = kInvalidId;
            uint32_t id_roughness = kInvalidId;
        };

        struct Conductor
        {
            uint32_t id_roughness_u = kInvalidId;
            uint32_t id_roughness_v = kInvalidId;
            uint32_t id_specular_reflectance = kInvalidId;
            Vec3 reflectivity = {};
            Vec3 edgetint = {};
        };

        struct Dielectric
        {
            uint32_t id_roughness_u = kInvalidId;
            uint32_t id_roughness_v = kInvalidId;
            uint32_t id_specular_reflectance = kInvalidId;
            uint32_t id_specular_transmittance = kInvalidId;
            float eta = 1.0f;
        };

        struct Plastic
        {
            float eta = 1.0f;
            uint32_t id_roughness = kInvalidId;
            uint32_t id_diffuse_reflectance = kInvalidId;
            uint32_t id_specular_reflectance = kInvalidId;
        };

        BSDF::Type type;
        bool twosided;
        uint32_t id_opacity;
        uint32_t id_bump_map;
        union
        {
            AreaLight area_light;
            Diffuse diffuse;
            RoughDiffuse rough_diffuse;
            Conductor conductor;
            Dielectric dielectric;
            Plastic plastic;
        };

        QUALIFIER_D_H Info();
        QUALIFIER_D_H ~Info() {}
        QUALIFIER_D_H Info(const Info &info);
        QUALIFIER_D_H void operator=(const Info &info);
    };

    struct SampleRec
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

    QUALIFIER_D_H BSDF();
    QUALIFIER_D_H BSDF(const uint32_t id, const BSDF::Info &info,
                       Texture *texture_buffer);

    QUALIFIER_D_H void Evaluate(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void Sample(uint32_t *seed, BSDF::SampleRec *rec) const;
    QUALIFIER_D_H Vec3 GetRadiance(const Vec2 &texcoord) const;
    QUALIFIER_D_H bool IsEmitter() const;
    QUALIFIER_D_H bool IsTwosided() const { return data_.twosided; }
    QUALIFIER_D_H bool IsTransparent(const Vec2 &texcoord,
                                     uint32_t *seed) const;

    QUALIFIER_D_H static float AverageFresnel(const float eta);

private:
    QUALIFIER_D_H void EvaluateDiffuse(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void EvaluateRoughDiffuse(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void EvaluateConductor(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void EvaluateDielectric(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void EvaluateThinDielectric(BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void EvaluatePlastic(BSDF::SampleRec *rec) const;

    QUALIFIER_D_H void SampleDiffuse(uint32_t *seed,
                                     BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void SampleRoughDiffuse(uint32_t *seed,
                                          BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void SampleConductor(uint32_t *seed,
                                       BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void SampleDielectric(uint32_t *seed,
                                        BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void SampleThinDielectric(uint32_t *seed,
                                            BSDF::SampleRec *rec) const;
    QUALIFIER_D_H void SamplePlastic(uint32_t *seed,
                                     BSDF::SampleRec *rec) const;

    template <typename T>
    QUALIFIER_D_H T FresnelSchlick(const float cos_theta,
                                   const T &relectivity) const
    {
        return (1.0f - relectivity) *
                   static_cast<float>(pow(1.0f - cos_theta, 5)) +
               relectivity;
    }

    uint32_t id_;
    Data data_;
};

} // namespace csrt
