#pragma once

#include "tensor.cuh"
#include "texture.cuh"
#include "utils.cuh"

namespace rt
{

struct SamplingRecord
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

    QUALIFIER_D_H Vec3 ToLocal(const Vec3 &v) const
    {
        return Normalize({Dot(v, tangent), Dot(v, bitangent), Dot(v, normal)});
    }

    QUALIFIER_D_H Vec3 ToWorld(const Vec3 &v) const
    {
        return Normalize(v.x * tangent + v.y * bitangent + v.z * normal);
    }
};

class Bsdf
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
            float reflectivity = 1.0f;
            float eta = 1.0f;
            float eta_inv = 1.0f;
        };

        struct Plastic
        {
            float reflectivity = 1.0f;
            float F_avg = 1.0f;
            uint32_t id_roughness = kInvalidId;
            uint32_t id_diffuse_reflectance = kInvalidId;
            uint32_t id_specular_reflectance = kInvalidId;
        };

        Bsdf::Type type;
        bool twosided;
        uint32_t id_opacity;
        uint32_t id_bump_map;
        Texture *texture_buffer;
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
        QUALIFIER_D_H Data(const Bsdf::Data &info);
        QUALIFIER_D_H void operator=(const Bsdf::Data &info);
    };

    struct Info
    {
        struct AreaLight
        {
            float weight = 1;
            uint32_t id_radiance = kInvalidId;
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

        Bsdf::Type type;
        bool twosided;
        uint32_t id_opacity;
        uint32_t id_bump_map;
        union
        {
            AreaLight area_light;
            Bsdf::Data::Diffuse diffuse;
            Bsdf::Data::RoughDiffuse rough_diffuse;
            Bsdf::Data::Conductor conductor;
            Dielectric dielectric;
            Plastic plastic;
        };

        Info();
        ~Info() {}
        Info(const Info &info);
        void operator=(const Info &info);

        static Bsdf::Info CreateAreaLight(const uint32_t id_radiance,
                                          const float weight,
                                          const bool twosided,
                                          const uint32_t id_opacity,
                                          const uint32_t id_bump_map);
        static Bsdf::Info CreateDiffuse(const uint32_t id_diffuse_reflectance,
                                        const bool twosided,
                                        const uint32_t id_opacity,
                                        const uint32_t id_bump_map);
        static Bsdf::Info CreateRoughDiffuse(
            const bool use_fast_approx, const uint32_t id_diffuse_reflectance,
            const uint32_t id_roughness, const bool twosided,
            const uint32_t id_opacity, const uint32_t id_bump_map);
        static Bsdf::Info
        CreateConductor(const uint32_t id_roughness_u,
                        const uint32_t id_roughness_v,
                        const uint32_t id_specular_reflectance, const Vec3 &eta,
                        const Vec3 &k, const bool twosided,
                        const uint32_t id_opacity, const uint32_t id_bump_map);
        static Bsdf::Info
        CreateDielectric(bool is_thin, const uint32_t id_roughness_u,
                         const uint32_t id_roughness_v,
                         const uint32_t id_specular_reflectance,
                         const uint32_t id_specular_transmittance,
                         const float eta, const bool twosided,
                         const uint32_t id_opacity, const uint32_t id_bump_map);
        static Bsdf::Info CreatePlastic(const float eta,
                                        const uint32_t id_roughness,
                                        const uint32_t id_diffuse_reflectance,
                                        const uint32_t id_specular_reflectance,
                                        const bool twosided,
                                        const uint32_t id_opacity,
                                        const uint32_t id_bump_map);
    };

    QUALIFIER_D_H Bsdf();
    QUALIFIER_D_H Bsdf(const uint32_t id, const Bsdf::Data &info);

    QUALIFIER_D_H void Evaluate(SamplingRecord *rec) const;
    QUALIFIER_D_H void Sample(const Vec3 &xi, SamplingRecord *rec) const;
    QUALIFIER_D_H Vec3 GetRadiance(const Vec2 &texcoord) const;
    QUALIFIER_D_H bool IsEmitter() const;
    QUALIFIER_D_H bool IsTwosided() const { return data_.twosided; }
    QUALIFIER_D_H bool IsTransparent(const Vec2 &texcoord,
                                     const float xi) const;

    static float AverageFresnel(const float eta);

private:
    QUALIFIER_D_H void EvaluateDiffuse(SamplingRecord *rec) const;
    QUALIFIER_D_H void EvaluateRoughDiffuse(SamplingRecord *rec) const;
    QUALIFIER_D_H void EvaluateConductor(SamplingRecord *rec) const;
    QUALIFIER_D_H void EvaluateDielectric(SamplingRecord *rec) const;
    QUALIFIER_D_H void EvaluateThinDielectric(SamplingRecord *rec) const;
    QUALIFIER_D_H void EvaluatePlastic(SamplingRecord *rec) const;

    QUALIFIER_D_H void SampleDiffuse(const Vec3 &xi, SamplingRecord *rec) const;
    QUALIFIER_D_H void SampleRoughDiffuse(const Vec3 &xi,
                                          SamplingRecord *rec) const;
    QUALIFIER_D_H void SampleConductor(const Vec3 &xi,
                                       SamplingRecord *rec) const;
    QUALIFIER_D_H void SampleDielectric(const Vec3 &xi,
                                        SamplingRecord *rec) const;
    QUALIFIER_D_H void SampleThinDielectric(const Vec3 &xi,
                                            SamplingRecord *rec) const;
    QUALIFIER_D_H void SamplePlastic(const Vec3 &xi, SamplingRecord *rec) const;

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

} // namespace rt
