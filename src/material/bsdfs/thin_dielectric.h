#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

class ThinDielectric : public Material
{
public:
    /**
     * \brief 薄的电介质材质
     * \param id 材质id
     * \param ext_ior 外折射率
     * \param int_ior 内折射率
     * \param specular_reflectance 可选参数，镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
     * \param specular_transmittance 可选参数，镜面透射系数。注意，对于物理真实感绘制，不应设置此参数。
     */
    ThinDielectric(const std::string &id,
                   Float ext_ior,
                   Float int_ior,
                   std::unique_ptr<Vector3> specular_reflectance = nullptr,
                   std::unique_ptr<Vector3> specular_transmittance = nullptr)
        : Material(id, MaterialType::kThinDielectric),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          specular_transmittance_(std::move(specular_transmittance)) {}

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
                bs.weight *= *specular_reflectance_;
            bs.pdf = kr;
        }
        else
        {
            bs.wi = wo;
            bs.weight = Vector3(1 - kr);
            if (specular_transmittance_)
                bs.weight *= *specular_transmittance_;
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
                weight *= *specular_reflectance_;
        }
        else if (SameDirection(wo, wi))
        {
            weight = Vector3(1 - kr);
            if (specular_transmittance_)
                weight *= *specular_transmittance_;
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

private:
    Float eta_inv_;                                   //光线射出材质的相对折射率
    std::unique_ptr<Vector3> specular_reflectance_;   //镜面反射系数。（注意：对于物理真实感绘制，不应设置此参数）
    std::unique_ptr<Vector3> specular_transmittance_; //镜面透射系数。（注意：对于物理真实感绘制，不应设置此参数）
};

NAMESPACE_END(simple_renderer)