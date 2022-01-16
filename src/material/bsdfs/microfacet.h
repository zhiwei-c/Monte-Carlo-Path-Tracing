#pragma once

#include "../material.h"
#include "../../utils/math/microfacet_distribution/microfacet_distributions.h"

NAMESPACE_BEGIN(simple_renderer)

constexpr int kResolution = 512;
constexpr Float step = 1.0 / kResolution;
constexpr int sample_count = 1024;
constexpr Float sample_count_inv = 1.0 / sample_count;

class Microfacet : public Material
{
public:
    /**
	 * \brief 微表面模型。参数 alpha 控制表面的粗糙程度：
     *      0.001 到 0.01 的 alpha 对应于有轻微瑕疵的光滑表面；
     *      0.1 的 alpha 是相对粗糙的表面；
     *      0.3 到 0.7 的 alpha 是极其粗糙的表面，例如被蚀刻或研磨的表面；
     *      过高的 alpha 可能不太接近实际情况。
	 * \param id 材质id
	 * \param type 材质类型
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha_u 沿切线（tangent）方向的粗糙度
	 * \param alpha_v 沿副切线（bitangent）方向的粗糙度
    */
    Microfacet(const std::string &id,
               MaterialType type,
               MicrofacetDistribType distrib_type,
               Float alpha_u,
               Float alpha_v)
        : Material(id, type),
          distrib_type_(distrib_type),
          alpha_u_(alpha_u),
          alpha_v_(alpha_v)
    {
        PrecomputeAlbedo();
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    virtual BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const = 0;

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    virtual Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const = 0;

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    virtual Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const = 0;

protected:
    MicrofacetDistribType distrib_type_; //用于模拟表面粗糙度的微表面分布的类型
    Float alpha_u_;                      //沿切线（tangent）方向的粗糙度
    Float alpha_v_;                      //沿副切线（bitangent）方向的粗糙度
    Float albedo_avg_;                   //平均反照率

    ///\brief 给定光线出射方向与法线方向夹角的余弦，获取反照率
    Float GetAlbedo(Float cos_theta) const
    {
        auto offset = cos_theta * kResolution;
        auto offset_int = static_cast<int>(offset);
        if (offset_int >= kResolution - 1)
            return albedo_[kResolution - 1];
        else
            return Lerp(offset - offset_int, albedo_[offset_int], albedo_[offset_int + 1]);
    }

private:
    Float albedo_[kResolution]; //光线出射方向与法线方向夹角的余弦从0到1的一系列反照率

    ///\brief 预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率和平均反照率
    void PrecomputeAlbedo()
    {
        auto normal = Vector3(0, 0, 1);
        auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);

        //预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率
        std::fill(albedo_, albedo_ + kResolution, 0);
        for (int j = 0; j < kResolution; j++)
        {
            Float cos_n_o = step * (static_cast<Float>(j) + 0.5);
            auto wo = Vector3(std::sqrt(1 - Sqr(cos_n_o)), 0, cos_n_o);
            for (int i = 0; i < sample_count; i++)
            {
                auto [normal_micro, pdf] = distrib->Sample(normal, Hammersley(i + 1, sample_count + 1));
                auto cos_m_o = std::max(glm::dot(wo, normal_micro),
                                        static_cast<Float>(0));
                auto cos_m_n = std::max(glm::dot(normal, normal_micro),
                                        static_cast<Float>(0));
                auto wi = -Reflect(-wo, normal_micro);
                auto G = distrib->SmithG1(-wi, normal_micro, normal) *
                         distrib->SmithG1(wo, normal_micro, normal);
                //重要性采样的微表面模型BSDF，并且菲涅尔项置为1（或0）
                albedo_[j] += (cos_m_o * G / (cos_n_o * cos_m_n));
            }
            albedo_[j] = sample_count_inv * albedo_[j];
        }

        albedo_avg_ = 0;
        //积分，计算平均反照率
        for (int j = 0; j < kResolution; j++)
        {
            Float cos_n_o = step * (static_cast<Float>(j) + 0.5);
            auto wo = Vector3(std::sqrt(1 - Sqr(cos_n_o)), 0, cos_n_o);
            Float avg_tmp = 0;
            for (int i = 0; i < sample_count; i++)
            {
                auto [normal_micro, pdf] = distrib->Sample(normal, Hammersley(i + 1, sample_count + 1));
                auto wi = -Reflect(-wo, normal_micro);
                auto cos_n_i = std::max(glm::dot(-wi, normal),
                                        static_cast<Float>(0));
                avg_tmp += (albedo_[j] * cos_n_i);
            }
            albedo_avg_ += (avg_tmp * 2 * sample_count_inv);
        }
        albedo_avg_ *= step;
    }
};

NAMESPACE_END(simple_renderer)