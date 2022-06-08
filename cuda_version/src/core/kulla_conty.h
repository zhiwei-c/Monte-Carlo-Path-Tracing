#pragma once

#include "../utils/math/sample.h"
#include "../utils/math/microfacet_distribution.h"

constexpr auto kAlbedoResolution = static_cast<int>(128);

__device__ inline vec3 HammersleyVec2(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    Float rdi = Float(bits) * 2.3283064365386963e-10;
    return vec3(Float(i) / Float(N), rdi, 0);
}

__device__ inline void CreateCosinAlbedoTexture(MicrofacetDistribType type, Texture *alpha_u_texture,
                                                Texture *alpha_v_texture, float *&albedo, float &albedo_avg)
{
    Float alpha_u = 0.1;
    if (alpha_u_texture)
    {
        if (alpha_u_texture->Varying())
            return;
        alpha_u = alpha_u_texture->Color(vec2(0)).x;
    }

    Float alpha_v = 0.1;
    if (alpha_v_texture)
    {
        if (alpha_v_texture->Varying())
            return;
        alpha_v = alpha_v_texture->Color(vec2(0)).x;
    }

    if (alpha_u < 0.01 && alpha_v < 0.01)
        return;

    constexpr auto resolution_step = static_cast<Float>(1.0 / kAlbedoResolution);
    constexpr auto sample_count = static_cast<int>(1024);
    constexpr auto sample_count_inv = static_cast<Float>(1.0 / sample_count);
    const auto macro_normal = vec3(0, 0, 1);
    //预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率
    albedo = new float[kAlbedoResolution];
    for (int j = 0; j < kAlbedoResolution; j++)
    {
        auto cos_n_o = (j + 0.5) * resolution_step;
        auto wo = vec3(sqrt(1.0 - cos_n_o * cos_n_o), 0, cos_n_o);
        for (int i = 0; i < sample_count; i++)
        {
            auto facet_normal = vec3(0);
            auto pdf = static_cast<Float>(0);
            SampleNormDistrib(type, alpha_u, alpha_v, macro_normal, HammersleyVec2(i + 1, sample_count + 1),
                              facet_normal, pdf);
            auto cos_m_o = glm::max(myvec::dot(wo, facet_normal), static_cast<Float>(0));
            auto cos_m_n = glm::max(myvec::dot(macro_normal, facet_normal), static_cast<Float>(0));
            auto wi = -Reflect(-wo, macro_normal);

            auto G = SmithG1(type, alpha_u, alpha_v, -wi, vec3(0, 0, 1), facet_normal) *
                     SmithG1(type, alpha_u, alpha_v, wo, vec3(0, 0, 1), facet_normal);
            //重要性采样的微表面模型BSDF，并且菲涅尔项置为1（或0）
            albedo[j] += (cos_m_o * G / (cos_n_o * cos_m_n));
        }
        albedo[j] = glm::min(albedo[j] * sample_count_inv, 1.0);
    }

    albedo_avg = 0;
    //积分，计算平均反照率
    for (int j = 0; j < kAlbedoResolution; j++)
    {
        auto cos_n_o = (j + 0.5) * resolution_step;
        auto wo = vec3(std::sqrt(1.0 - cos_n_o * cos_n_o), 0, cos_n_o);
        Float avg_tmp = 0;
        for (int i = 0; i < sample_count; i++)
        {
            auto facet_normal = vec3(0);
            auto pdf = static_cast<Float>(0);
            SampleNormDistrib(type, alpha_u, alpha_v, macro_normal, HammersleyVec2(i + 1, sample_count + 1),
                              facet_normal, pdf);
            auto wi = -Reflect(-wo, facet_normal);
            auto cos_n_i = glm::max(myvec::dot(-wi, macro_normal), static_cast<Float>(0));
            avg_tmp += (albedo[j] * cos_n_i);
        }
        albedo_avg += (avg_tmp * 2.0 * sample_count_inv);
    }
    albedo_avg *= resolution_step;

    if (albedo_avg > 1 - kEpsilon)
    {
        albedo_avg = -1;
        delete[] albedo;
        albedo = nullptr;
    }
}