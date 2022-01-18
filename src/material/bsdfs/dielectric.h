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
               std::unique_ptr<Spectrum> specular_reflectance = nullptr,
               std::unique_ptr<Spectrum> specular_transmittance = nullptr)
        : Material(id, MaterialType::kDielectric),
          eta_(int_ior / ext_ior),
          eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)),
          specular_transmittance_(std::move(specular_transmittance)) {}

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta = inside ? eta_ : eta_inv_;     //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        BsdfSampling bs;
        auto angle = std::acos(glm::dot(wo, normal)) * kPiInv * 180;
        auto kr = Fresnel(-wo, normal, eta_inv);
        auto r_0 = Sqr((eta_inv - 1) / (eta_inv + 1));
        auto kr_tmp = r_0 + (1 - r_0) * std::pow(1 - glm::dot(wo, normal), 5);
        auto sample_x = UniformFloat();
        if (sample_x < kr)
        {
            bs.wi = -Reflect(-wo, normal);

            bs.weight = Spectrum(kr);

            bs.pdf = kr;
        }
        else
        {
            bs.wi = -Refract(-wo, normal, eta_inv);

            bs.weight = Spectrum(1 - kr);
            if (specular_transmittance_)
                bs.weight *= *specular_transmittance_;
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            bs.weight *= Sqr(eta);

            bs.pdf = 1 - kr;
        }

        if (bs.pdf < kEpsilonL)
            return BsdfSampling();

        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto kr = Fresnel(wi, normal, eta_inv);
        if (SameDirection(wo, Reflect(wi, normal)))
        {
            auto albedo = Spectrum(kr);
            if (specular_reflectance_)
                albedo *= *specular_reflectance_;
            return albedo;
        }
        else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
        {
            auto weight = Spectrum(1 - kr);
            if (specular_transmittance_)
                weight *= *specular_transmittance_;
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            weight *= Sqr(eta_inv);
            return weight;
        }
        else
            return Spectrum(0);
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

        auto kr = Fresnel(wi, normal, eta_inv);

        if (SameDirection(wo, Reflect(wi, normal)))
            return kr;
        else if (kr > kOneMinusEpsilon)
            return 0;
        else if (SameDirection(wo, Refract(wi, normal, eta_inv)))
            return 1 - kr;
        else
            return 0;
    }

private:
    Float eta_;                                        //光线射入材质的相对折射率
    Float eta_inv_;                                    //光线从材质内部射出的相对折射率
    std::unique_ptr<Spectrum> specular_reflectance_;   //调节镜面反射分量的可选参数。（注意：对于物理真实感绘制，不应设置此参数）
    std::unique_ptr<Spectrum> specular_transmittance_; //调节镜面透射分量的可选参数。（注意：对于物理真实感绘制，不应设置此参数）
};

NAMESPACE_END(simple_renderer)