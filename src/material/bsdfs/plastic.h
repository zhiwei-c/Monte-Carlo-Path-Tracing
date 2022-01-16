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
	 * \param diffuse_reflectance 可选参数，漫反射系数
	 * \param diffuse_map  漫反射纹理
	 * \param nonlinear 是否考虑因内部散射而引起的非线性色移
	 * \param ext_ior 外折射率
	 * \param int_ior 内折射率
	 * \param specular_reflectance 可选参数，镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
	*/
    Plastic(const std::string &id,
            const Spectrum &diffuse_reflectance,
            Texture *diffuse_map,
            bool nonlinear,
            Float ext_ior,
            Float int_ior,
            const Spectrum &specular_reflectance = Spectrum(1))
        : Material(id, MaterialType::kPlastic),
          diffuse_reflectance_(diffuse_reflectance),
          diffuse_map_(diffuse_map),
          nonlinear_(nonlinear),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(specular_reflectance)
    {
        fdr_int_ = FresnelDiffuseReflectance(eta_inv_);
        fdr_ext_ = FresnelDiffuseReflectance(eta_);
        auto d_sum = diffuse_reflectance_.r + diffuse_reflectance_.g + diffuse_reflectance_.b;
        s_sum_ = specular_reflectance_.r + specular_reflectance_.g + specular_reflectance_.b;
        specular_sampling_weight_ = s_sum_ / (d_sum + s_sum_);
    }

    ~Plastic()
    {
        if (diffuse_map_)
            DeleteTexturePointer(diffuse_map_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto specular_sampling_weight = specular_sampling_weight_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto kd = diffuse_map_->GetPixel(*texcoord);
                auto d_sum = kd.r + kd.g + kd.b;
                specular_sampling_weight = s_sum_ / (d_sum + s_sum_);
            }
        }

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
        if (bs.pdf < kEpsilon)
            return BsdfSampling();
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
        auto diffuse_reflectance = diffuse_reflectance_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
                diffuse_reflectance = diffuse_map_->GetPixel(*texcoord);
        }
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
        {
            albedo += kr_i * specular_reflectance_;
        }

        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
        auto kr = Fresnel(wi, normal, eta_inv);
        auto specular_sampling_weight = specular_sampling_weight_;
        if (texcoord != nullptr)
        {
            if (diffuse_map_)
            {
                auto kd = diffuse_map_->GetPixel(*texcoord);
                auto d_sum = kd.r + kd.g + kd.b;
                specular_sampling_weight = s_sum_ / (d_sum + s_sum_);
            }
        }

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
    Spectrum diffuse_reflectance_;  // 漫反射系数，
    Texture *diffuse_map_;          // 漫反射纹理
    bool nonlinear_;                // 是否考虑因内部散射而引起的非线性色移
    Float eta_;                     //光线射入材质的相对折射率
    Float eta_inv_;                 //光线射出材质的相对折射率
    Spectrum specular_reflectance_; // 镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。

    Float fdr_ext_;
    Float fdr_int_;
    Float s_sum_;
    Float specular_sampling_weight_;
};

NAMESPACE_END(simple_renderer)