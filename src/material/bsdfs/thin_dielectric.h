#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

class ThinDielectric : public Material
{
public:
    /**
     * \brief 薄的电介质材质
     * \param id 材质id
     * \param int_ior 内折射率
     * \param ext_ior 外折射率
     * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
     * \param specular_transmittance 镜面透射系数。注意，对于物理真实感绘制，应默认为 1。
     */
    ThinDielectric(const std::string &id,
                   Float int_ior,
                   Float ext_ior,
                   Texture *specular_reflectance = nullptr,
                   Texture *specular_transmittance = nullptr)
        : Material(id, MaterialType::kThinDielectric),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(specular_reflectance),
          specular_transmittance_(specular_transmittance) {}

    ~ThinDielectric()
    {
        if (specular_reflectance_)
        {
            delete specular_reflectance_;
            specular_reflectance_ = nullptr;
        }
        if (specular_transmittance_)
        {
            delete specular_transmittance_;
            specular_transmittance_ = nullptr;
        }
    }
    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
    {
        auto kr = Fresnel(-wo, normal, eta_inv_);

        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2 / (1 + kr);

        BsdfSampling bs;
        auto sample_x = UniformFloat();
        if (sample_x < kr)
        {
            bs.wi = -Reflect(-wo, normal);
            bs.weight = Vector3(kr);
            if (specular_reflectance_)
            {
                if (texcoord != nullptr)
                    bs.weight *= specular_reflectance_->GetPixel(*texcoord);
                else
                    bs.weight *= specular_reflectance_->GetPixel(Vector2(0));
            }
            bs.pdf = kr;
        }
        else
        {
            bs.wi = wo;
            bs.weight = Vector3(1 - kr);
            if (specular_transmittance_)
            {
                if (texcoord != nullptr)
                    bs.weight *= specular_transmittance_->GetPixel(*texcoord);
                else
                    bs.weight *= specular_transmittance_->GetPixel(Vector2(0));
            }
            bs.pdf = 1 - kr;
        }
        if (bs.pdf < kEpsilon)
            return BsdfSampling();
        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        Vector3 weight(0);

        auto kr = Fresnel(wi, normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2 / (1 + kr);

        if (SameDirection(wo, Reflect(wi, normal)))
        {
            weight = Vector3(kr);
            if (specular_reflectance_)
            {
                if (texcoord != nullptr)
                    weight *= specular_reflectance_->GetPixel(*texcoord);
                else
                    weight *= specular_reflectance_->GetPixel(Vector2(0));
            }
        }
        else if (SameDirection(wo, wi))
        {
            weight = Vector3(1 - kr);
            if (specular_transmittance_)
            {
                if (texcoord != nullptr)
                    weight *= specular_transmittance_->GetPixel(*texcoord);
                else
                    weight *= specular_transmittance_->GetPixel(Vector2(0));
            }
        }
        return weight;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto kr = Fresnel(wi, normal, eta_inv_);
        //考虑光线在材质内部多次反射: r' = r + trt + tr^3t + ..
        if (kr < 1)
            kr *= 2 / (1 + kr);

        if (SameDirection(wo, Reflect(wi, normal)))
        {
            return kr;
        }
        else if (SameDirection(wo, wi))
        {
            return 1 - kr;
        }
        else
            return 0;
    }

    bool TextureMapping() const override { return (specular_reflectance_ && !specular_reflectance_->Constant()) ||
                                                  (specular_transmittance_ && !specular_transmittance_->Constant()); }

private:
    Float eta_inv_;                   //光线射出材质的相对折射率
    Texture *specular_reflectance_;   //镜面反射系数。（注意：对于物理真实感绘制，应默认为 1）
    Texture *specular_transmittance_; //镜面透射系数。（注意：对于物理真实感绘制，应默认为 1）
};

NAMESPACE_END(simple_renderer)