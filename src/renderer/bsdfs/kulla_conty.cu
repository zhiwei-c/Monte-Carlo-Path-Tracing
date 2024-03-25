#include "csrt/renderer/bsdfs/kulla_conty.cuh"

#include "csrt/rtcore/ray.cuh"
#include "csrt/utils.cuh"

namespace
{

using namespace csrt;

float IntegrateBRDF(const Vec3 &V, const float roughness)
{
    constexpr uint32_t sample_count = 1024;
    constexpr float step = 1.0f / sample_count;
    const Vec3 N = {0.0f, 0.0f, 1.0f};

    float pdf_h, brdf_accum = 0.0f;
    Vec3 H, L;
    for (uint32_t i = 0; i < sample_count; ++i)
    {
        SampleGgx(i * step, GetVanDerCorputSequence<2>(i), roughness, &H,
                  &pdf_h);
        L = Ray::Reflect(V, H);
        const float G = SmithG1Ggx(roughness, -V, H) *
                        SmithG1Ggx(roughness, L, H),
                    N_dot_V = Dot(N, -V), N_Dot_L = Dot(N, L),
                    N_dot_H = Dot(N, H), H_dot_V = Dot(H, -V);
        if (N_Dot_L > 0.0f && N_dot_H > 0.0f && H_dot_V > 0.0f)
            brdf_accum += (H_dot_V * G) / (N_dot_V * N_dot_H);
    }
    return fminf(brdf_accum * step, 1.0f);
}

float IntegrateAlbedo(const Vec3 &V, const float roughness, const float brdf)
{
    constexpr uint32_t sample_count = 1024;
    constexpr float step = 1.0f / sample_count;
    const Vec3 N = {0.0f, 0.0f, 1.0f};

    float pdf_h, albedo_accum = 0.0f;
    Vec3 H, L;
    for (uint32_t i = 0; i < sample_count; ++i)
    {
        SampleGgx(i * step, GetVanDerCorputSequence<2>(i), roughness, &H,
                  &pdf_h);
        L = Ray::Reflect(V, H);

        const float N_Dot_L = Dot(N, L), N_dot_H = Dot(N, H),
                    H_dot_V = Dot(-V, H);
        if (N_Dot_L > 0.0f && N_dot_H > 0.0f && H_dot_V > 0.0f)
            albedo_accum += brdf * N_Dot_L;
    }
    return albedo_accum * 2.0f * step;
}

} // namespace

namespace csrt
{

void ComputeKullaConty(float *brdf_buffer, float *albedo_avg_buffer)
{
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

QUALIFIER_D_H float GetBrdfAvg(float *brdf_avg_buffer, const float cos_theta,
                               const float roughness)
{
    const float offset1 = roughness * kLutResolution,
                offset2 = cos_theta * kLutResolution;
    const int offset_int1 = static_cast<int>(offset1),
              offset_int2 = static_cast<int>(offset2);
    if (offset_int1 >= kLutResolution - 1)
    {
        if (offset_int2 >= kLutResolution - 1)
        {
            return brdf_avg_buffer[(kLutResolution - 1) * kLutResolution +
                                   kLutResolution - 1];
        }
        else
        {
            return Lerp(brdf_avg_buffer[(kLutResolution - 1) * kLutResolution +
                                        offset_int2],
                        brdf_avg_buffer[(kLutResolution - 1) * kLutResolution +
                                        offset_int2 + 1],
                        offset2 - offset_int2);
        }
    }
    else
    {
        if (offset_int2 >= kLutResolution - 1)
        {
            return Lerp(brdf_avg_buffer[offset_int1 * kLutResolution +
                                        kLutResolution - 1],
                        brdf_avg_buffer[(offset_int1 + 1) * kLutResolution +
                                        kLutResolution - 1],
                        offset1 - offset_int1);
        }
        else
        {
            return Lerp(
                Lerp(
                    brdf_avg_buffer[offset_int1 * kLutResolution + offset_int2],
                    brdf_avg_buffer[(offset_int1 + 1) * kLutResolution +
                                    offset_int2],
                    offset1 - offset_int1),
                Lerp(brdf_avg_buffer[offset_int1 * kLutResolution +
                                     offset_int2 + 1],
                     brdf_avg_buffer[(offset_int1 + 1) * kLutResolution +
                                     offset_int2 + 1],
                     offset1 - offset_int1),
                offset2 - offset_int2);
        }
    }
}

QUALIFIER_D_H float GetAlbedoAvg(float *albedo_avg_buffer,
                                 const float roughness)
{
    const float offset = roughness * kLutResolution;
    const int offset_int = static_cast<int>(offset);
    if (offset_int >= kLutResolution - 1)
        return albedo_avg_buffer[kLutResolution - 1];
    else
        return Lerp(albedo_avg_buffer[offset_int],
                    albedo_avg_buffer[offset_int + 1], offset - offset_int);
}

} // namespace csrt