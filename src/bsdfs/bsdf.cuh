#pragma once

#include <string>

#include "../tensor/tensor.cuh"
#include "../textures/texture.cuh"
#include "../renderer/ray.cuh"

struct SamplingRecord
{
    bool valid = false;
    bool inside = false; // 表面法线方向是否朝向表面内侧
    float pdf = 0.0f;    // 光线从该方向入射的概率
    Vec2 texcoord;       // 纹理坐标
    Vec3 wi;             // 光线入射方向
    Vec3 wo;             // 光线出射方向
    Vec3 pos;            // 表面位置
    Vec3 normal;         // 表面法线方向
    Vec3 attenuation;    // BSDF 光能衰减系数
};

class Bsdf
{
public:
    static constexpr uint64_t kLutResolution = 128;

    QUALIFIER_DEVICE static void ComputeKullaConty(float *brdf, float *albedo_avg);

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
            uint64_t id_opacity;
            uint64_t id_bumpmap;

            struct AreaLight
            {
                uint64_t id_radiance;
            } area_light;

            struct Diffuse
            {
                uint64_t id_diffuse_reflectance;
            } diffuse;

            struct RoughDiffuse
            {
                bool use_fast_approx;
                uint64_t id_diffuse_reflectance;
                uint64_t id_roughness;
            } rough_diffuse;

            struct Conductor
            {
                uint64_t id_roughness;
                uint64_t id_specular_reflectance;
                Vec3 reflectivity;
                Vec3 edgetint;
            } conductor;

            struct Dielectric
            {
                uint64_t id_roughness;
                uint64_t id_specular_reflectance;
                uint64_t id_specular_transmittance;
                float eta;
            } dielectric;

            struct Plastic
            {
                uint64_t id_roughness;
                uint64_t id_diffuse_reflectance;
                uint64_t id_specular_reflectance;
                float eta;
            } plastic;
        } data;

        static Info CreateAreaLight(const uint64_t id_radiance, const bool twosided,
                                    const uint64_t id_opacity, const uint64_t id_bumpmap);
        static Info CreateDiffuse(const uint64_t id_diffuse_reflectance, const bool twosided,
                                  const uint64_t id_opacity, const uint64_t id_bumpmap);
        static Info CreateRoughDiffuse(const uint64_t id_diffuse_reflectance,
                                       const uint64_t id_roughness, const bool use_fast_approx,
                                       const bool twosided, const uint64_t id_opacity,
                                       const uint64_t id_bumpmap);
        static Info CreateConductor(const uint64_t id_roughness, const uint64_t id_specular_reflectance,
                                    const Vec3 &eta, const Vec3 &k, const bool twosided,
                                    const uint64_t id_opacity, const uint64_t id_bumpmap);
        static Info CreateDielectric(const uint64_t id_roughness, const uint64_t id_specular_reflectance,
                                     const uint64_t id_specular_transmittance, const float eta,
                                     bool is_thin_dielectric, const bool twosided,
                                     const uint64_t id_opacity, const uint64_t id_bumpmap);
        static Info CreatePlastic(const uint64_t id_roughness, const uint64_t id_diffuse_reflectance,
                                  const uint64_t id_specular_reflectance, const float eta,
                                  const bool twosided, const uint64_t id_opacity,
                                  const uint64_t id_bumpmap);
    };

    QUALIFIER_DEVICE virtual ~Bsdf() {}

    /// @brief 根据光线的方向评估能量的衰减，法线方向被调整为与入射光线同侧
    QUALIFIER_DEVICE virtual void Evaluate(const float *pixel_buffer, Texture **texture_buffer,
                                           uint64_t *seed, SamplingRecord *rec) const {}

    /// @brief 根据出射光线抽样入射光线，法线方向被调整为与出射光线同侧
    QUALIFIER_DEVICE virtual void Sample(const float *pixel_buffer, Texture **texture_buffer,
                                         uint64_t *seed, SamplingRecord *rec) const {}

    QUALIFIER_DEVICE virtual Vec3 GetRadiance(const Vec2 &texcoord, const float *pixel_buffer,
                                              Texture **texture_buffer) const { return Vec3(0); }

    QUALIFIER_DEVICE Vec3 ApplyBumpMapping(const Vec3 &normal, const Vec3 &tangent,
                                           const Vec3 &bitangent, const Vec2 &texcoord,
                                           const float *pixel_buffer, Texture **texture_buffer,
                                           uint64_t *seed) const;

    QUALIFIER_DEVICE void SetKullaConty(float *brdf, float *albedo_avg);

    QUALIFIER_DEVICE bool HasEmission() const { return type_ == Type::kAreaLight; }
    QUALIFIER_DEVICE bool IsTwosided() const { return twosided_; }
    QUALIFIER_DEVICE bool IsTransparent(const Vec2 &texcoord, const float *pixel_buffer,
                                        Texture **texture_buffer, uint64_t *seed) const;

protected:
    QUALIFIER_DEVICE Bsdf(const uint64_t id, const Type type, bool twosided, const uint64_t id_opacity,
                          const uint64_t id_bumpmap)
        : id_(id), type_(type), twosided_(twosided), brdf_(nullptr), albedo_avg_(nullptr),
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
    uint64_t id_;
    uint64_t id_opacity_;
    uint64_t id_bumpmap_;
};

class AreaLight : public Bsdf
{
public:
    QUALIFIER_DEVICE AreaLight(const uint64_t id, const Bsdf::Info::Data &data)
        : Bsdf(id, kAreaLight, data.twosided, data.id_opacity, data.id_bumpmap),
          id_radiance_(data.area_light.id_radiance)
    {
    }

    QUALIFIER_DEVICE Vec3 GetRadiance(const Vec2 &texcoord, const float *pixel_buffer,
                                      Texture **texture_buffer) const override
    {
        return texture_buffer[id_radiance_]->GetColor(texcoord, pixel_buffer);
    }

private:
    uint64_t id_radiance_;
};