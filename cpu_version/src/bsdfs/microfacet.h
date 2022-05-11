#pragma once

#include <memory>
#include <utility>
#include <array>

#include "../core/texture.h"
#include "../core/microfacet_distribution.h"

NAMESPACE_BEGIN(raytracer)

constexpr int kResolution = 512; //预计算纹理贴图的精度
constexpr Float step = 1.0 / kResolution;
constexpr int sample_count = 1024;
constexpr Float sample_count_inv = 1.0 / sample_count;

///\brief 微表面模型基类
class Microfacet
{
public:
    /**
     * \brief 微表面模型。参数 alpha 控制表面的粗糙程度：
     *      0.001 到 0.01 的 alpha 对应于有轻微瑕疵的光滑表面；
     *      0.1 的 alpha 是相对粗糙的表面；
     *      0.3 到 0.7 的 alpha 是极其粗糙的表面，例如被蚀刻或研磨的表面；
     *      过高的 alpha 可能不太接近实际情况。
     * \param type 材质类型
     * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
     * \param alpha_u 沿切线（tangent）方向的粗糙度
     * \param alpha_v 沿副切线（bitangent）方向的粗糙度
     */
    Microfacet(MicrofacetDistribType distrib_type, std::unique_ptr<Texture> alpha_u, std::unique_ptr<Texture> alpha_v)
        : distrib_type_(distrib_type), albedo_avg_(-1), alpha_u_(std::move(alpha_u)), alpha_v_(std::move(alpha_v))
    {
    }

protected:
    MicrofacetDistribType distrib_type_; //微表面分布类型
    std::unique_ptr<Texture> alpha_u_;   //沿切线（tangent）方向的粗糙度
    std::unique_ptr<Texture> alpha_v_;   //沿副切线（bitangent）方向的粗糙度
    Float albedo_avg_;                   //平均反照率

    ///\brief 给定表面纹理坐标，获取该点粗糙程度
    std::pair<Float, Float> GetAlpha(const Vector2 &texcoord) const
    {
        Float alpha_u = 0, alpha_v = 0;
        alpha_u = alpha_u_->Color(texcoord).x;
        alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : alpha_u;
        return {alpha_u, alpha_v};
    }

    ///\brief 给定光线出射方向与法线方向夹角的余弦，获取反照率
    Float GetAlbedo(Float cos_theta) const
    {
        Float offset = cos_theta * kResolution;
        auto offset_int = static_cast<int>(offset);
        if (offset_int >= kResolution - 1)
            return albedo_.back();
        else
            return Lerp(offset - offset_int, albedo_[offset_int], albedo_[offset_int + 1]);
    }

    ///\brief 预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率和平均反照率
    void ComputeAlbedoTable()
    {
        albedo_.fill(0);
        auto normal = Vector3(0, 0, 1);
        auto [alpha_u, alpha_v] = GetAlpha(Vector2(0));
        if (alpha_u < 0.01 || alpha_v < 0.01)
            return;
        auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
        //预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率
        for (size_t j = 0; j < kResolution; j++)
        {
            Float cos_theta_o = step * (j + 0.5);
            auto wo = Vector3(std::sqrt(1 - Sqr(cos_theta_o)), 0, cos_theta_o);
            for (int i = 0; i < sample_count; i++)
            {
                auto [normal_micro, pdf] = distrib->Sample(normal, Hammersley(i + 1, sample_count + 1));
                Vector3 wi = -Reflect(-wo, normal_micro);
                Float G = distrib->SmithG1(-wi, normal_micro, normal) *
                          distrib->SmithG1(wo, normal_micro, normal),
                      cos_m_o = std::max(glm::dot(wo, normal_micro), 0.0),
                      cos_m_n = std::max(glm::dot(normal, normal_micro), 0.0);
                //重要性采样的微表面模型BSDF，并且菲涅尔项置为1（或0）
                albedo_[j] += (cos_m_o * G / (cos_theta_o * cos_m_n));
            }
            albedo_[j] = std::max(sample_count_inv * albedo_[j], 0.0);
        }

        albedo_avg_ = 0;
        //积分，计算平均反照率
        for (size_t j = 0; j < kResolution; j++)
        {
            Float avg_tmp = 0, cos_theta_o = step * (j + 0.5);
            auto wo = Vector3(std::sqrt(1.0 - Sqr(cos_theta_o)), 0, cos_theta_o);
            for (int i = 0; i < sample_count; i++)
            {
                auto [normal_micro, pdf] = distrib->Sample(normal, Hammersley(i + 1, sample_count + 1));
                Vector3 wi = -Reflect(-wo, normal_micro);
                Float cos_theta_i = std::max(glm::dot(-wi, normal), static_cast<Float>(0));
                avg_tmp += (albedo_[j] * cos_theta_i);
            }
            albedo_avg_ += (avg_tmp * 2 * sample_count_inv);
        }
        albedo_avg_ *= step;
        if (albedo_avg_ > 1.0 - kEpsilon)
            albedo_avg_ = -1;
    }

private:
    std::array<Float, kResolution> albedo_; //光线出射方向与法线方向夹角的余弦从0到1的一系列反照率
};

NAMESPACE_END(raytracer)