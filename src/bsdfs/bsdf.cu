#include "bsdf.cuh"

#include <cassert>
#include <cstdio>

#include "../utils/math.cuh"
#include "ior.cuh"

Bsdf::Info Bsdf::Info::CreateAreaLight(const uint64_t id_radiance, const bool twosided,
                                       const uint64_t id_opacity, const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kAreaLight;
    info.data.area_light.id_radiance = id_radiance;
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

Bsdf::Info Bsdf::Info::CreateDiffuse(const uint64_t id_diffuse_reflectance, const bool twosided,
                                     const uint64_t id_opacity, const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kDiffuse;
    info.data.diffuse.id_diffuse_reflectance = id_diffuse_reflectance;
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

Bsdf::Info Bsdf::Info::CreateRoughDiffuse(const uint64_t id_diffuse_reflectance,
                                          const uint64_t id_roughness, const bool use_fast_approx,
                                          const bool twosided, const uint64_t id_opacity,
                                          const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kRoughDiffuse;
    info.data.rough_diffuse.id_diffuse_reflectance = id_diffuse_reflectance;
    info.data.rough_diffuse.id_roughness = id_roughness;
    info.data.rough_diffuse.use_fast_approx = use_fast_approx;
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

Bsdf::Info Bsdf::Info::CreateConductor(const uint64_t id_roughness,
                                       const uint64_t id_specular_reflectance, const Vec3 &eta,
                                       const Vec3 &k, const bool twosided, const uint64_t id_opacity,
                                       const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kConductor;
    info.data.conductor.id_roughness = id_roughness;
    info.data.conductor.id_specular_reflectance = id_specular_reflectance;
    info.data.conductor.reflectivity = (Sqr(eta - 1.0f) + Sqr(k)) / (Sqr(eta + 1.0f) + Sqr(k));
    const Vec3 temp1 = 1.0f + Sqrt(info.data.conductor.reflectivity),
               temp2 = 1.0f - Sqrt(info.data.conductor.reflectivity),
               temp3 = ((1.0f - info.data.conductor.reflectivity) /
                        (1.0 + info.data.conductor.reflectivity));
    info.data.conductor.edgetint = (temp1 - eta * temp2) / (temp1 - temp3 * temp2);
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

Bsdf::Info Bsdf::Info::CreateDielectric(const uint64_t id_roughness,
                                        const uint64_t id_specular_reflectance,
                                        const uint64_t id_specular_transmittance, const float eta,
                                        bool is_thin_dielectric, const bool twosided,
                                        const uint64_t id_opacity, const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = is_thin_dielectric ? Bsdf::Type::kThinDielectric : Bsdf::Type::kDielectric;
    info.data.dielectric.id_roughness = id_roughness;
    info.data.dielectric.id_specular_reflectance = id_specular_reflectance;
    info.data.dielectric.id_specular_transmittance = id_specular_transmittance;
    info.data.dielectric.eta = eta;
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

Bsdf::Info Bsdf::Info::CreatePlastic(const uint64_t id_roughness, const uint64_t id_diffuse_reflectance,
                                     const uint64_t id_specular_reflectance, const float eta,
                                     const bool twosided, const uint64_t id_opacity,
                                     const uint64_t id_bumpmap)
{
    Bsdf::Info info;
    info.type = Bsdf::Type::kPlastic;
    info.data.plastic.id_roughness = id_roughness;
    info.data.plastic.id_specular_reflectance = id_specular_reflectance;
    info.data.plastic.id_diffuse_reflectance = id_diffuse_reflectance;
    info.data.plastic.eta = eta;
    info.data.twosided = twosided;
    info.data.id_opacity = id_opacity;
    info.data.id_bumpmap = id_bumpmap;
    return info;
}

QUALIFIER_DEVICE void Bsdf::ComputeKullaConty(float *brdf_buffer, float *albedo_avg_buffer)
{
    auto IntegrateBRDF = [](const Vec3 &V, const float roughness)
    {
        constexpr uint32_t sample_count = 1024;
        constexpr float step = 1.0f / sample_count;
        const Vec3 N = {0.0f, 0.0f, 1.0f};

        float pdf_h, brdf_accum = 0.0f;
        Vec3 H, L;
        for (uint32_t i = 0; i < sample_count; ++i)
        {
            SampleGgx(i * step, GetVanDerCorputSequence(i, 2), roughness, H, pdf_h);
            L = Reflect(V, H);
            const float G = SmithG1Ggx(roughness, -V, N, H) * SmithG1Ggx(roughness, L, N, H),
                        N_dot_V = Dot(N, -V),
                        N_Dot_L = Dot(N, L),
                        N_dot_H = Dot(N, H),
                        H_dot_V = Dot(H, -V);
            if (N_Dot_L > 0.0f && N_dot_H > 0.0f && H_dot_V > 0.0f)
                brdf_accum += (H_dot_V * G) / (N_dot_V * N_dot_H);
        }
        return fminf(brdf_accum * step, 1.0f);
    };

    auto IntegrateAlbedo = [](const Vec3 &V, const float roughness, const float brdf)
    {
        constexpr uint32_t sample_count = 1024;
        constexpr float step = 1.0f / sample_count;
        const Vec3 N = {0.0f, 0.0f, 1.0f};

        float pdf_h, albedo_accum = 0.0f;
        Vec3 H, L;
        for (uint32_t i = 0; i < sample_count; ++i)
        {
            SampleGgx(i * step, GetVanDerCorputSequence(i, 2), roughness, H, pdf_h);
            L = Reflect(V, H);

            const float N_Dot_L = Dot(N, L),
                        N_dot_H = Dot(N, H),
                        H_dot_V = Dot(-V, H);
            if (N_Dot_L > 0.0f && N_dot_H > 0.0f && H_dot_V > 0.0f)
                albedo_accum += brdf * N_Dot_L;
        }
        return albedo_accum * 2.0f * step;
    };

    float step = 1.0f / kLutResolution, albedo_accum = 0.0f;
    for (int i = kLutResolution - 1; i >= 0; --i)
    {
        albedo_accum = 0.0f;
        float roughness = step * (static_cast<float>(i) + 0.5f);
        for (int j = kLutResolution - 1; j >= 0; --j)
        {
            const float N_dot_V = step * (static_cast<float>(j) + 0.5f);
            const Vec3 V = {-sqrtf(1.f - N_dot_V * N_dot_V), 0.0f, -N_dot_V};
            const float brdf_avg = IntegrateBRDF(V, roughness);

            brdf_buffer[i * kLutResolution + j] = brdf_avg;
            albedo_accum += IntegrateAlbedo(V, roughness, brdf_avg);
        }
        albedo_avg_buffer[i] = albedo_accum * step;
    }
}

QUALIFIER_DEVICE Vec3 Bsdf::ApplyBumpMapping(const Vec3 &normal, const Vec3 &tangent,
                                             const Vec3 &bitangent, const Vec2 &texcoord,
                                             const float *pixel_buffer, Texture **texture_buffer,
                                             uint64_t *seed) const
{
    if (id_bumpmap_ == kInvalidId)
        return normal;

    const Vec2 gradient = texture_buffer[id_bumpmap_]->GetGradient(texcoord, pixel_buffer);
    const Vec3 normal_pertubed = Normalize(-gradient.u * tangent - gradient.v * bitangent + normal);
    return normal_pertubed;
}

QUALIFIER_DEVICE void Bsdf::SetKullaConty(float *brdf, float *albedo_avg)
{
    brdf_ = brdf, albedo_avg_ = albedo_avg;
}

QUALIFIER_DEVICE bool Bsdf::IsTransparent(const Vec2 &texcoord, const float *pixel_buffer,
                                          Texture **texture_buffer, uint64_t *seed) const
{
    return (id_opacity_ != kInvalidId &&
            texture_buffer[id_opacity_]->IsTransparent(texcoord, pixel_buffer, seed));
}

QUALIFIER_DEVICE float Bsdf::GetAlbedoAvg(const float roughness) const
{
    const float offset = roughness * kLutResolution;
    const int offset_int = static_cast<int>(offset);
    if (offset_int >= kLutResolution - 1)
    {
        return albedo_avg_[kLutResolution - 1];
    }
    else
    {
        return Lerp(albedo_avg_[offset_int], albedo_avg_[offset_int + 1], offset - offset_int);
    }
}

QUALIFIER_DEVICE float Bsdf::GetBrdfAvg(const float cos_theta, const float roughness) const
{
    const float offset1 = roughness * kLutResolution,
                offset2 = cos_theta * kLutResolution;
    const int offset_int1 = static_cast<int>(offset1),
              offset_int2 = static_cast<int>(offset2);
    if (offset_int1 >= kLutResolution - 1)
    {
        if (offset_int2 >= kLutResolution - 1)
        {
            return brdf_[(kLutResolution - 1) * kLutResolution + kLutResolution - 1];
        }
        else
        {
            return Lerp(brdf_[(kLutResolution - 1) * kLutResolution + offset_int2],
                        brdf_[(kLutResolution - 1) * kLutResolution + offset_int2 + 1],
                        offset2 - offset_int2);
        }
    }
    else
    {
        if (offset_int2 >= kLutResolution - 1)
        {
            return Lerp(brdf_[offset_int1 * kLutResolution + kLutResolution - 1],
                        brdf_[(offset_int1 + 1) * kLutResolution + kLutResolution - 1],
                        offset1 - offset_int1);
        }
        else
        {
            return Lerp(Lerp(brdf_[offset_int1 * kLutResolution + offset_int2],
                             brdf_[(offset_int1 + 1) * kLutResolution + offset_int2],
                             offset1 - offset_int1),
                        Lerp(brdf_[offset_int1 * kLutResolution + offset_int2 + 1],
                             brdf_[(offset_int1 + 1) * kLutResolution + offset_int2 + 1],
                             offset1 - offset_int1),
                        offset2 - offset_int2);
        }
    }
}
