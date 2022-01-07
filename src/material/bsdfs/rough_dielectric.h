#pragma once

#include "../material.h"
#include "../../utils/math/microfacet_distribution/microfacet_distributions.h"

NAMESPACE_BEGIN(simple_renderer)

class RoughDielectric : public Material
{
public:
	/**
	 * \brief 粗糙的电介质材质
	 * \param id 材质id
	 * \param ior_ext 外折射率
	 * \param int_ior 内折射率
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha_u 沿切线（tangent）方向的粗糙度
	 * \param alpha_v 沿副切线（bitangent）方向的粗糙度
	 * \param specular_reflectance 可选参数，镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
	 * \param specular_transmittance 可选参数，镜面透射系数。注意，对于物理真实感绘制，不应设置此参数。
	*/
	RoughDielectric(const std::string &id,
					Float ior_ext,
					Float ior_int,
					MicrofacetDistribType distrib_type,
					Float alpha_u,
					Float alpha_v,
					std::unique_ptr<Vector3> specular_reflectance = nullptr,
					std::unique_ptr<Vector3> specular_transmittance = nullptr)
		: Material(id, MaterialType::kRoughDielectric),
		  ior_in_(ior_int / ior_ext),
		  ior_out_(ior_ext / ior_int),
		  distrib_type_(distrib_type),
		  alpha_u_(alpha_u),
		  alpha_v_(alpha_v),
		  specular_reflectance_(std::move(specular_reflectance)),
		  specular_transmittance_(std::move(specular_transmittance)) {}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
	{
		const auto &wi_pseudo = -wo;
		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);

		// Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，
		// 略微缩放粗糙度，以减少重要性采样权重。
		auto cos_i_n = glm::dot(wi_pseudo, normal);
		distrib->ScaleAlpha(1.2 - 0.2 * std::sqrt(std::fabs(cos_i_n)));

		auto normal_micro = distrib->Sample(normal);
		DeleteDistribPointer(distrib);

		auto eta_inv = inside ? ior_in_ : ior_out_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
		auto kr_pseudo = Fresnel(wi_pseudo, normal_micro, eta_inv);

		auto sample_x = UniformFloat();
		if (sample_x < kr_pseudo)
			return {-Reflect(wi_pseudo, normal_micro), BsdfSamplingType::kReflection};
		else
			return {-Refract(wi_pseudo, normal_micro, eta_inv), BsdfSamplingType::kTransmission};
	}

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
	{
		auto eta_inv = inside ? ior_in_ : ior_out_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

		auto cos_o_n = glm::dot(wo, normal);
		auto cos_i_n = glm::dot(-wi, normal);

		Vector3 h(0);
		Float F = 0;
		auto relfect = cos_o_n > 0;
		if (relfect)
		{
			h = glm::normalize(-wi + wo);
			F = Fresnel(wi, h, eta_inv);
		}
		else
		{
			h = glm::normalize(-eta_inv * wi + wo);
			if (NotSameHemis(h, normal))
				h = -h;
			F = Fresnel(wi, h, eta_inv);
		}

		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);
		auto D = distrib->Eval(h, normal);
		auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);
		DeleteDistribPointer(distrib);

		if (relfect)
		{
			auto weight = Vector3(F * D * G / (4 * std::fabs(cos_i_n * cos_o_n)));
			if (specular_reflectance_)
				weight *= *specular_reflectance_;
			return weight;
		}
		else
		{
			auto cos_i_h = glm::dot(-wi, h),
				 cos_o_h = glm::dot(wo, h);
			auto weight = Vector3(std::fabs(cos_i_h * cos_o_h * (1 - F) * G * D /
											(cos_i_n * cos_o_n * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (specular_transmittance_)
				weight *= *specular_transmittance_;

			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			weight *= Sqr(eta_inv);

			return weight;
		}
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
	{
		auto eta_inv = inside ? ior_in_ : ior_out_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

		auto cos_i_n = glm::dot(-wi, normal),
			 cos_o_n = glm::dot(wo, normal);

		Vector3 h(0);
		bool relfect = cos_o_n > 0;
		if (relfect)
		{
			h = glm::normalize(-wi + wo);
		}
		else
		{
			h = glm::normalize(-eta_inv * wi + wo);
			if (NotSameHemis(h, normal))
				h = -h;
		}

		auto F = Fresnel(wi, h, eta_inv);

		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);
		auto D = distrib->Eval(h, normal);
		DeleteDistribPointer(distrib);

		if (relfect)
		{
			auto jacobian = std::fabs(1 / (4 * glm::dot(wo, h)));
			return F * D * jacobian;
		}
		else
		{
			auto jacobian = std::fabs(glm::dot(wo, h) /
									  Sqr(eta_inv * glm::dot(-wi, h) + glm::dot(wo, h)));
			return (1 - F) * D * jacobian;
		}
	}

private:
	Float ior_in_;									  //光线射入材质的相对折射率
	Float ior_out_;									  //光线射出材质的相对折射率
	MicrofacetDistribType distrib_type_;			  // 用于模拟表面粗糙度的微表面分布的类型
	Float alpha_u_;									  // 沿切线（tangent）方向的粗糙度
	Float alpha_v_;									  // 沿副切线（bitangent）方向的粗糙度
	std::unique_ptr<Vector3> specular_reflectance_;	  // 镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。
	std::unique_ptr<Vector3> specular_transmittance_; // 镜面透射系数。注意，对于物理真实感绘制，不应设置此参数。
};

NAMESPACE_END(simple_renderer)