#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

class Dielectric : public Material
{
public:
    /**
	 * \brief 光滑的电介质材质
	 * \param id 材质id
	 * \param ext_ior 外折射率
	 * \param int_ior 内折射率
	 * \param specular_reflectance 可选参数，调节镜面反射分量。注意，对于物理真实感绘制，不应设置此参数。
	 * \param specular_transmittance 可选参数，调节镜面透射分量。注意，对于物理真实感绘制，不应设置此参数。
	*/
    Dielectric(const std::string &id,
               Float ext_ior,
               Float int_ior,
               std::unique_ptr<Vector3> specular_reflectance = nullptr,
               std::unique_ptr<Vector3> specular_transmittance = nullptr)
        : Material(id, MaterialType::kDielectric),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          specular_transmittance_(std::move(specular_transmittance)) {}

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        const auto &wi_pseudo = -wo;
        auto kr_pseudo = Fresnel(wi_pseudo, normal, eta_inv);
        auto sample_x = UniformFloat();
        if (sample_x < kr_pseudo)
            return {-Reflect(wi_pseudo, normal), BsdfSamplingType::kReflection};
        else
            return {-Refract(wi_pseudo, normal, eta_inv), BsdfSamplingType::kTransmission};
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        Vector3 weight(0);
        auto kr = Fresnel(wi, normal, eta_inv);
        if (bsdf_sampling_type == BsdfSamplingType::kReflection ||
            SameDirection(wo, Reflect(wi, normal)))
        {
            weight = Vector3(kr);
            if (specular_reflectance_)
                weight *= *specular_reflectance_;
        }
        else if (bsdf_sampling_type == BsdfSamplingType::kTransmission ||
                 SameDirection(wo, Refract(wi, normal, eta_inv)))
        {
            weight = Vector3(1 - kr);
            if (specular_transmittance_)
                weight *= *specular_transmittance_;

            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            weight *= Sqr(eta_inv);
        }
        return weight;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto kr = Fresnel(wi, normal, eta_inv);

        if (bsdf_sampling_type == BsdfSamplingType::kReflection ||
            SameDirection(wo, Reflect(wi, normal)))
            return kr;

        if (FloatEqual(kr, 1))
            return 0;

        if (bsdf_sampling_type == BsdfSamplingType::kTransmission ||
            SameDirection(wo, Refract(wi, normal, eta_inv)))
            return 1 - kr;

        return 0;
    }

private:
    Float eta_;                                       //光线射入材质的相对折射率
    Float eta_inv_;                                   //光线从材质内部射出的相对折射率
    std::unique_ptr<Vector3> specular_reflectance_;   //调节镜面反射分量的可选参数。（注意：对于物理真实感绘制，不应设置此参数）
    std::unique_ptr<Vector3> specular_transmittance_; //调节镜面透射分量的可选参数。（注意：对于物理真实感绘制，不应设置此参数）
};

NAMESPACE_END(simple_renderer)