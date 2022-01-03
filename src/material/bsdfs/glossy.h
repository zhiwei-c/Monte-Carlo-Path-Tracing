#pragma once

#include "../material.h"

#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class Glossy : public Material
{
public:
    /**
     * \brief 冯模型定义的有光泽的材质
     * \param id 材质id
     * \param diffuse_reflectance 漫反射系数
     * \param specular_reflectance 镜面反射系数
     * \param exponent 镜面反射指数系数
     * \param diffuse_map 漫反射纹理
     */
    Glossy(const std::string &id,
           const Vector3 &diffuse_reflectance,
           const Vector3 &specular_reflectance,
           Float exponent,
           Texture *diffuse_map)
        : Material(id, MaterialType::kGlossy),
          diffuse_reflectance_(diffuse_reflectance),
          specular_reflectance_(specular_reflectance),
          exponent_(exponent),
          diffuse_map_(diffuse_map)
    {
        diffuse_reflectance_sum_ = diffuse_reflectance.r + diffuse_reflectance.g + diffuse_reflectance.b;
        specular_reflectance_sum_ = specular_reflectance.r + specular_reflectance.g + specular_reflectance.b;
    }

    ~Glossy()
    {
        if (diffuse_map_)
            DeleteTexturePointer(diffuse_map_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto diffuse_reflectance_sum = diffuse_reflectance_sum_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto diffuse_reflectance = diffuse_map_->GetPixel(*texcoord);
                diffuse_reflectance_sum = diffuse_reflectance.r + diffuse_reflectance.g + diffuse_reflectance.b;
            }
        }
        auto thresh = diffuse_reflectance_sum / (diffuse_reflectance_sum + specular_reflectance_sum_);
        auto sample_x = UniformFloat();
        if (sample_x < thresh)
        {
            auto wo_pseudo_local = HemisCos();
            return {-ToWorld(wo_pseudo_local, normal), BsdfSamplingType::kReflection};
        }
        else
        {
            auto wo_pseudo_local = HemisCosN(exponent_);
            auto wr_pseudo = Reflect(-wo, normal);
            auto wi = -ToWorld(wo_pseudo_local, wr_pseudo);
            if (SameHemis(wi, normal))
                return {Vector3(0), BsdfSamplingType::kNone};
            else
                return {wi, BsdfSamplingType::kReflection};
        }
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {
        // 入射、出射光线需在同侧
        if (NotSameHemis(wo, normal))
            return Vector3(0);

        Vector3 weight(0);
        // 计算漫反射分量的贡献
        if (texcoord != nullptr && diffuse_map_)
            weight += diffuse_map_->GetPixel(*texcoord) * kPiInv;
        else
            weight += diffuse_reflectance_ * kPiInv;
        // 计算镜面反射分量的贡献
        auto wr = Reflect(wi, normal);
        auto cos_alpha = glm::dot(wr, wo);
        if (cos_alpha > kEpsilon)
            weight += specular_reflectance_ * static_cast<Float>((exponent_ + 2) * kPiInv * 0.5 * std::pow(cos_alpha, exponent_));

        return weight;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {
        if (bsdf_sampling_type == BsdfSamplingType::kNone ||
            NotSameHemis(wo, normal))
            return 0;

        auto diffuse_reflectance_sum = diffuse_reflectance_sum_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto diffuse_reflectance = diffuse_map_->GetPixel(*texcoord);
                diffuse_reflectance_sum = diffuse_reflectance.r + diffuse_reflectance.g + diffuse_reflectance.b;
            }
        }

        auto thresh = diffuse_reflectance_sum / (diffuse_reflectance_sum + specular_reflectance_sum_);

        auto wr = Reflect(wi, normal);
        if (NotSameHemis(wo, wr))
        {
            auto wo_local = ToLocal(wo, normal);
            return thresh * PdfHemisCos(wo_local);
        }
        else
        {
            auto sample_x = UniformFloat();
            if (sample_x < thresh)
            {
                auto wo_local = ToLocal(wo, normal);
                return thresh * PdfHemisCos(wo_local);
            }
            else
            {
                auto wo_local = ToLocal(wo, wr);
                return (1 - thresh) * PdfHemisCosN(wo_local, exponent_);
            }
        }
    }

    bool TextureMapping() const override { return diffuse_map_ != nullptr; }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else if (diffuse_map_)
            return diffuse_map_->Transparent(texcoord);
        else
            return false;
    }

private:
    Vector3 diffuse_reflectance_;  //漫反射系数
    Texture *diffuse_map_;         //漫反射纹理
    Vector3 specular_reflectance_; //镜面反射系数
    Float exponent_;               //镜面反射指数系数

    Float diffuse_reflectance_sum_;
    Float specular_reflectance_sum_;
};

NAMESPACE_END(simple_renderer)