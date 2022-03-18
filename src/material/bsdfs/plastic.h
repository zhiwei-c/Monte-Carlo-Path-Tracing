#pragma once

#include "../material.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

class Plastic : public Material
{
public:
    /**
     * \brief 光滑的塑料材质
     * \param id 材质id
     * \param int_ior 内折射率
     * \param ext_ior 外折射率
     * \param diffuse_reflectance 漫反射系数
     * \param nonlinear 是否考虑因内部散射而引起的非线性色移
     * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
     */
    Plastic(const std::string &id,
            Float int_ior,
            Float ext_ior,
            std::unique_ptr<Texture> diffuse_reflectance,
            bool nonlinear,
            std::unique_ptr<Texture> specular_reflectance = nullptr)
        : Material(id, MaterialType::kPlastic),
          diffuse_reflectance_(std::move(diffuse_reflectance)),
          nonlinear_(nonlinear),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance))
    {
        fdr_int_ = FresnelDiffuseReflectance(eta_inv_);
        fdr_ext_ = FresnelDiffuseReflectance(eta_);
        specular_sampling_weight_ = 0;

        if (!diffuse_reflectance_->Constant() || (specular_reflectance_ && !specular_reflectance_->Constant()))
            return;

        auto d_weight = diffuse_reflectance_->GetPixel(Vector2(0));
        auto d_sum = d_weight.r + d_weight.g + d_weight.b;
        Float s_sum = 3;
        if (specular_reflectance_)
        {
            auto ks = specular_reflectance_->GetPixel(Vector2(0));
            s_sum = ks.r + ks.g + ks.b;
        }

        specular_sampling_weight_ = s_sum / (d_sum + s_sum);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto specular_sampling_weight = get_specular_sampling_weight(texcoord);

        auto kr = Fresnel(-wo, normal, eta_inv);
        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1 - kr) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

        BsdfSampling bs;
        auto sample_x = UniformFloat();
        if (sample_x < pdf_specular)
        {
            bs.wi = -Reflect(-wo, normal);
        }
        else
        {
            auto [wi_local, pdf] = HemisCos();
            bs.wi = -ToWorld(wi_local, normal);
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
        if (NotSameHemis(wo, normal))
            return Spectrum(0);

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto fdr_int = !inside ? fdr_int_ : fdr_ext_;

        Spectrum albedo(0);
        auto diffuse_reflectance = get_diffuse_reflectance(texcoord);
        if (nonlinear_)
        {
            albedo = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_int);
        }
        else
        {
            albedo = diffuse_reflectance / (1 - fdr_int);
        }

        auto kr_i = Fresnel(wi, normal, eta_inv);
        auto kr_o = Fresnel(-wo, normal, eta_inv);
        albedo *= Sqr(eta_inv) * (1 - kr_i) * (1 - kr_o) * kPiInv;

        if (SameDirection(Reflect(wi, normal), wo))
            albedo += kr_i * get_specular_reflectance(texcoord);

        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto kr = Fresnel(wi, normal, eta_inv);
        auto specular_sampling_weight = get_specular_sampling_weight(texcoord);

        auto pdf_specular = kr * specular_sampling_weight,
             pdf_diffuse = (1 - kr) * (1 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        pdf_diffuse = 1 - pdf_specular;

        auto wo_local = ToLocal(wo, normal);
        auto pdf = PdfHemisCos(wo_local);
        auto result = pdf_diffuse * pdf;

        if (SameDirection(wo, Reflect(wi, normal)))
        {
            result += pdf_specular;
        }
        return result;
    }

    bool TextureMapping() const override { return (specular_reflectance_ && !specular_reflectance_->Constant()) || !diffuse_reflectance_->Constant(); }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else
            return diffuse_reflectance_->Transparent(texcoord);
    }

private:
    bool nonlinear_; // 是否考虑因内部散射而引起的非线性色移
    Float eta_;      // 光线射入材质的相对折射率
    Float eta_inv_;  // 光线射出材质的相对折射率
    Float fdr_ext_;
    Float fdr_int_;
    Float specular_sampling_weight_;
    std::unique_ptr<Texture> specular_reflectance_; // 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
    std::unique_ptr<Texture> diffuse_reflectance_;  // 漫反射系数

    Spectrum get_diffuse_reflectance(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant())
            return diffuse_reflectance_->GetPixel(Vector2(0));
        else
            return diffuse_reflectance_->GetPixel(*texcoord);
    }
    Spectrum get_specular_reflectance(const Vector2 *texcoord) const
    {
        if (!diffuse_reflectance_)
            return Spectrum(1);
        if (diffuse_reflectance_->Constant())
            return diffuse_reflectance_->GetPixel(Vector2(0));
        else
            return diffuse_reflectance_->GetPixel(*texcoord);
    }

    Float get_specular_sampling_weight(const Vector2 *texcoord) const
    {
        if (diffuse_reflectance_->Constant() && (!specular_reflectance_ || specular_reflectance_->Constant()))
            return specular_sampling_weight_;

        auto kd = diffuse_reflectance_->GetPixel(*texcoord);
        auto d_sum = kd.r + kd.g + kd.b;
        if (!specular_reflectance_)
            return 3 / (d_sum + 3);

        auto s_weight = specular_reflectance_->GetPixel(*texcoord);
        auto s_sum = s_weight.r + s_weight.g + s_weight.b;
        return s_sum / (d_sum + s_sum);
    }
};

NAMESPACE_END(simple_renderer)