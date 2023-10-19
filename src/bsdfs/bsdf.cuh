#pragma once

#include <string>

#include "../tensor/tensor.cuh"
#include "../textures/texture.cuh"
#include "../renderer/ray.cuh"

struct SamplingRecord
{
    bool valid;
    bool inside;
    float pdf;
    Vec2 texcoord;
    Vec3 wi;
    Vec3 wo;
    Vec3 position;
    Vec3 normal;
    Vec3 tangent;
    Vec3 bitangent;
    Vec3 attenuation;
    QUALIFIER_DEVICE SamplingRecord()
        : valid(false), inside(false), pdf(0), texcoord(Vec2()), wi(Vec3()), wo(Vec3()),
          position(Vec3()), tangent(Vec3()), bitangent(bitangent)
    {
    }
};

class Bsdf
{
public:
    enum Type
    {
        kAreaLight,
        kConductor,
        kDielectric,
        kDiffuse,
        kRoughDiffuse,
        kPlastic,
        kThinDielectric,
    };

    struct Info
    {
        Type type;
        struct Data
        {
            bool twosided;
            uint32_t id_opacity;
            uint32_t id_bumpmap;

            struct AreaLight
            {
                uint32_t id_radiance;
            } area_light;

            struct Diffuse
            {
                uint32_t id_diffuse_reflectance;
            } diffuse;

            struct RoughDiffuse
            {
                bool use_fast_approx;
                uint32_t id_diffuse_reflectance;
                uint32_t id_roughness;
            } rough_diffuse;

            struct Conductor
            {
                uint32_t id_roughness;
                uint32_t id_specular_reflectance;
                Vec3 reflectivity;
                Vec3 edgetint;
            } conductor;

            struct Dielectric
            {
                uint32_t id_roughness;
                uint32_t id_specular_reflectance;
                uint32_t id_specular_transmittance;
                float eta;
            } dielectric;

            struct Plastic
            {
                uint32_t id_roughness;
                uint32_t id_diffuse_reflectance;
                uint32_t id_specular_reflectance;
                float eta;
            } plastic;
        } data;

        static Info CreateAreaLight(const uint32_t id_radiance, const bool twosided,
                                    const uint32_t id_opacity, const uint32_t id_bumpmap);
        static Info CreateDiffuse(const uint32_t id_diffuse_reflectance, const bool twosided,
                                  const uint32_t id_opacity, const uint32_t id_bumpmap);
        static Info CreateRoughDiffuse(const uint32_t id_diffuse_reflectance,
                                       const uint32_t id_roughness, const bool use_fast_approx,
                                       const bool twosided, const uint32_t id_opacity,
                                       const uint32_t id_bumpmap);
        static Info CreateConductor(const uint32_t id_roughness,
                                    const uint32_t id_specular_reflectance,
                                    const Vec3 &eta, const Vec3 &k, const bool twosided,
                                    const uint32_t id_opacity, const uint32_t id_bumpmap);
        static Info CreateDielectric(const uint32_t id_roughness,
                                     const uint32_t id_specular_reflectance,
                                     const uint32_t id_specular_transmittance, const float eta,
                                     bool is_thin_dielectric, const bool twosided,
                                     const uint32_t id_opacity, const uint32_t id_bumpmap);
        static Info CreatePlastic(const uint32_t id_roughness,
                                  const uint32_t id_diffuse_reflectance,
                                  const uint32_t id_specular_reflectance, const float eta,
                                  const bool twosided, const uint32_t id_opacity,
                                  const uint32_t id_bumpmap);
    };

    static constexpr uint32_t kLutResolution = 128;

    QUALIFIER_DEVICE virtual ~Bsdf() {}

    /// @brief 根据光线的方向评估能量的衰减，法线方向被调整为与入射光线同侧
    QUALIFIER_DEVICE virtual void Evaluate(Texture **texture_buffer, const float *pixel_buffer,
                                           uint32_t *seed, SamplingRecord *rec) const
    {
    }

    /// @brief 根据出射光线抽样入射光线，法线方向被调整为与出射光线同侧
    QUALIFIER_DEVICE virtual void Sample(Texture **texture_buffer, const float *pixel_buffer,
                                         uint32_t *seed, SamplingRecord *rec) const
    {
    }

    QUALIFIER_DEVICE virtual Vec3 GetRadiance(const Vec2 &texcoord, Texture **texture_buffer,
                                              const float *pixel_buffer) const
    {
        return Vec3(0);
    }

    QUALIFIER_DEVICE Vec3 ApplyBumpMapping(const Vec3 &normal, const Vec3 &tangent,
                                           const Vec3 &bitangent, const Vec2 &texcoord,
                                           Texture **texture_buffer, const float *pixel_buffer,
                                           uint32_t *seed) const;

    QUALIFIER_DEVICE void SetKullaConty(float *brdf, float *albedo_avg);

    QUALIFIER_DEVICE bool HasEmission() const { return type_ == Type::kAreaLight; }
    QUALIFIER_DEVICE bool IsTwosided() const { return twosided_; }
    QUALIFIER_DEVICE bool IsTransparent(const Vec2 &texcoord, Texture **texture_buffer,
                                        const float *pixel_buffer, uint32_t *seed) const;

    QUALIFIER_DEVICE static void ComputeKullaConty(float *brdf, float *albedo_avg);

protected:
    QUALIFIER_DEVICE Bsdf(const uint32_t id_bsdf, const Type type, bool twosided,
                          const uint32_t id_opacity, const uint32_t id_bumpmap)
        : id_bsdf_(id_bsdf), type_(type), twosided_(twosided), brdf_(nullptr), albedo_avg_(nullptr),
          id_opacity_(id_opacity), id_bumpmap_(id_bumpmap)
    {
    }

    QUALIFIER_DEVICE float GetAlbedoAvg(const float roughness) const;
    QUALIFIER_DEVICE float GetBrdfAvg(const float cos_theta, const float roughness) const;

private:
    Type type_;
    float *brdf_;
    float *albedo_avg_;
    bool twosided_;
    uint32_t id_bsdf_;
    uint32_t id_opacity_;
    uint32_t id_bumpmap_;
};

class AreaLight : public Bsdf
{
public:
    QUALIFIER_DEVICE AreaLight(const uint32_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kAreaLight, data.twosided, data.id_opacity, data.id_bumpmap),
          id_radiance_(data.area_light.id_radiance)
    {
    }

    QUALIFIER_DEVICE Vec3 GetRadiance(const Vec2 &texcoord, Texture **texture_buffer,
                                      const float *pixel_buffer) const override
    {
        return texture_buffer[id_radiance_]->GetColor(texcoord, pixel_buffer);
    }

private:
    uint32_t id_radiance_;
};