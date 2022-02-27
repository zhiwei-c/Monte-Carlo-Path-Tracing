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
           Texture *diffuse_reflectance,
           Texture *specular_reflectance,
           Float exponent)
        : Material(id, MaterialType::kGlossy),
          diffuse_reflectance_(diffuse_reflectance),
          specular_reflectance_(specular_reflectance),
          exponent_(exponent)
    {
        diffuse_reflectance_sum_ = 0,
        specular_reflectance_sum_ = 0;
        if (diffuse_reflectance_->Constant())
        {
            auto kd = diffuse_reflectance->GetPixel(Vector2(0));
            diffuse_reflectance_sum_ = kd.r + kd.g + kd.b;
        }
        if (specular_reflectance->Constant())
        {
            auto ks = specular_reflectance->GetPixel(Vector2(0));
            specular_reflectance_sum_ = ks.r + ks.g + ks.b;
        }
        if (diffuse_reflectance_->Constant() && specular_reflectance->Constant())
            diffuse_sampling_weight_ = diffuse_reflectance_sum_ / (diffuse_reflectance_sum_ + specular_reflectance_sum_);
    }

    ~Glossy()
    {
        delete diffuse_reflectance_;
        diffuse_reflectance_ = nullptr;
        delete specular_reflectance_;
        specular_reflectance_ = nullptr;
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
    {
        BsdfSampling bs;
        auto pdf_diffuse = get_diffuse_sampling_weight(texcoord);
        auto sample_x = UniformFloat();
        if (sample_x < pdf_diffuse)
        {
            auto [wi_local, pdf] = HemisCos();
            bs.wi = -ToWorld(wi_local, normal);
        }
        else
        {
            auto wi_local = HemisCosN(exponent_);
            auto wr_pseudo = Reflect(-wo, normal);
            auto wi = -ToWorld(wi_local, wr_pseudo);
            if (SameHemis(wi, normal))
                return BsdfSampling();
        }
        bs.pdf = Pdf(bs.wi, wo, normal, texcoord, inside);
        if (bs.pdf < kEpsilonL)
            return BsdfSampling();

        if (get_weight)
            bs.weight = Eval(bs.wi, wo, normal, texcoord, inside);

        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        // 入射、出射光线需在同侧
        if (NotSameHemis(wo, normal))
            return Spectrum(0);

        Spectrum albedo(0);
        // 计算漫反射分量的贡献
        albedo += get_diffuse_reflectance(texcoord) * kPiInv;
        // 计算镜面反射分量的贡献
        auto wr = Reflect(wi, normal);
        auto cos_alpha = glm::dot(wr, wo);
        if (cos_alpha > kEpsilon)
            albedo += get_specular_reflectance(texcoord) * static_cast<Float>((exponent_ + 2) * kPiInv * 0.5 * std::pow(cos_alpha, exponent_));

        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto pdf_diffuse = get_diffuse_sampling_weight(texcoord);

        auto wr = Reflect(wi, normal);
        if (NotSameHemis(wo, wr))
        {
            auto wo_local = ToLocal(wo, normal);
            return pdf_diffuse * PdfHemisCos(wo_local);
        }
        else
        {
            auto sample_x = UniformFloat();
            if (sample_x < pdf_diffuse)
            {
                auto wo_local = ToLocal(wo, normal);
                return pdf_diffuse * PdfHemisCos(wo_local);
            }
            else
            {
                auto wo_local = ToLocal(wo, wr);
                return (1 - pdf_diffuse) * PdfHemisCosN(wo_local, exponent_);
            }
        }
    }

    bool TextureMapping() const override { return !diffuse_reflectance_->Constant() || !specular_reflectance_->Constant(); }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else
            return diffuse_reflectance_->Transparent(texcoord);
    }

private:
    Texture *diffuse_reflectance_;  //漫反射系数
    Texture *specular_reflectance_; //镜面反射系数
    Float exponent_;                //镜面反射指数系数

    Float diffuse_reflectance_sum_;
    Float specular_reflectance_sum_;
    Float diffuse_sampling_weight_;

    Spectrum get_specular_reflectance(const Vector2 *texcoord) const
    {
        if (specular_reflectance_->Constant())
            return specular_reflectance_->GetPixel(Vector2(0));
        else
            return specular_reflectance_->GetPixel(*texcoord);
    }

    Spectrum get_diffuse_reflectance(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant())
            return diffuse_reflectance_->GetPixel(Vector2(0));
        else
            return diffuse_reflectance_->GetPixel(*texcoord);
    }

    Float get_diffuse_sampling_weight(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant() && specular_reflectance_)
            return diffuse_sampling_weight_;

        auto ks_sum = specular_reflectance_sum_;
        if (!specular_reflectance_->Constant())
        {
            auto ks = get_specular_reflectance(texcoord);
            ks_sum = ks.r + ks.g + ks.b;
        }

        auto kd_sum = diffuse_reflectance_sum_;
        if (!diffuse_reflectance_->Constant())
        {
            auto kd = get_diffuse_reflectance(texcoord);
            kd_sum = kd.r + kd.g + kd.b;
        }

        return kd_sum / (kd_sum + ks_sum);
    }
};

NAMESPACE_END(simple_renderer)