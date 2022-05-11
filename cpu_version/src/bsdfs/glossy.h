#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 冯模型描述的有光泽材质派生类
class Glossy : public Material
{
public:
    ///\brief 冯模型描述的有光泽材质
    ///\param diffuse_reflectance 漫反射系数
    ///\param specular_reflectance 镜面反射系数
    ///\param exponent 镜面反射指数系数
    Glossy(std::unique_ptr<Texture> diffuse_reflectance,
           std::unique_ptr<Texture> specular_reflectance,
           Float exponent)
        : Material(MaterialType::kGlossy),
          diffuse_reflectance_(std::move(diffuse_reflectance)),
          specular_reflectance_(std::move(specular_reflectance)),
          exponent_(exponent), diffuse_sampling_weight_(-1),
          diffuse_reflectance_sum_(-1), specular_reflectance_sum_(-1)
    {
        if (diffuse_reflectance_->Constant())
        {
            Spectrum kd = diffuse_reflectance->Color(Vector2(0));
            diffuse_reflectance_sum_ = kd.r + kd.g + kd.b;
        }
        if (specular_reflectance->Constant())
        {
            Spectrum ks = specular_reflectance->Color(Vector2(0));
            specular_reflectance_sum_ = ks.r + ks.g + ks.b;
        }
        if (diffuse_reflectance_->Constant() && specular_reflectance->Constant())
            diffuse_sampling_weight_ = diffuse_reflectance_sum_ / (diffuse_reflectance_sum_ + specular_reflectance_sum_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        Float pdf_diffuse = DiffuseSamplingWeight(bs.texcoord);
        if (UniformFloat() < pdf_diffuse) //抽样漫反射分量
            SampleHemisCos(bs.normal, bs.wi);
        else
        { //抽样镜面反射反射分量
            bs.wi = SampleHemisCosN(exponent_, bs.normal);
            if (SameHemis(bs.wi, bs.normal))
                return;
        }
        bs.pdf = Pdf(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
        if (bs.get_attenuation)
            bs.attenuation = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        auto albedo = Spectrum(0);
        // 计算漫反射分量的贡献
        albedo += diffuse_reflectance_->Color(texcoord) * kPiInv;
        // 计算镜面反射分量的贡献
        Vector3 wr = Reflect(wi, normal);
        Float cos_alpha = glm::dot(wr, wo);
        if (cos_alpha > kEpsilon)
            albedo += specular_reflectance_->Color(texcoord) *
                      static_cast<Float>((exponent_ + 2) * kPiInv * 0.5 * std::pow(cos_alpha, exponent_));
        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        Float pdf_diffuse = DiffuseSamplingWeight(texcoord),
              pdf = pdf_diffuse * PdfHemisCos(wo, normal);
        Vector3 wr = Reflect(wi, normal);
        if (SameHemis(wo, wr))
            pdf += (1.0 - pdf_diffuse) * PdfHemisCosN(wo, wr, exponent_);

        return pdf;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               !diffuse_reflectance_->Constant() ||
               !specular_reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Material::Transparent(texcoord) ||
               diffuse_reflectance_->Transparent(texcoord);
    }

private:
    ///\brief 获取给定点抽样漫反射的权重
    Float DiffuseSamplingWeight(const Vector2 &texcoord) const
    {
        if (diffuse_sampling_weight_ >= 0)
            return diffuse_sampling_weight_;

        Float ks_sum = specular_reflectance_sum_;
        if (ks_sum < 0)
        {
            Spectrum ks = specular_reflectance_->Color(texcoord);
            ks_sum = ks.r + ks.g + ks.b;
        }

        Float kd_sum = diffuse_reflectance_sum_;
        if (kd_sum < 0)
        {
            Spectrum kd = diffuse_reflectance_->Color(texcoord);
            kd_sum = kd.r + kd.g + kd.b;
        }

        return kd_sum / (kd_sum + ks_sum);
    }

    Float exponent_;                                //镜面反射指数系数
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数
    Float diffuse_reflectance_sum_;                 //漫反射系数和
    Float specular_reflectance_sum_;                //镜面反射系数和
    Float diffuse_sampling_weight_;                 //抽样漫反射权重
};

NAMESPACE_END(raytracer)