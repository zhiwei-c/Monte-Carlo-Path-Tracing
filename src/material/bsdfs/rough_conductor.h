#pragma once

#include <algorithm>

#include "microfacet.h"

NAMESPACE_BEGIN(simple_renderer)

class RoughConductor : public Microfacet
{
public:
	/**
	 * \brief 粗糙的导体材质
	 * \param id 材质id
	 * \param mirror 是否是镜面（全反射）
	 * \param eta 材质折射率的实部
	 * \param k 材质折射率的虚部（消光系数）
	 * \param specular_reflectance 可选参数，镜面反射系数。注意，对于物理真实感绘制，不应设置此参数
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha_u 沿切线（tangent）方向的粗糙度
	 * \param alpha_v 沿副切线（bitangent）方向的粗糙度
	*/
	RoughConductor(const std::string &id,
				   bool mirror,
				   const Vector3 &eta,
				   const Vector3 &k,
				   MicrofacetDistribType distrib_type,
				   Float alpha_u,
				   Float alpha_v,
				   Float ext_ior = IOR.at("air"),
				   std::unique_ptr<Vector3> specular_reflectance = nullptr)
		: Microfacet(id,
					 MaterialType::kRoughConductor,
					 distrib_type,
					 alpha_u,
					 alpha_v),
		  mirror_(mirror),
		  eta_(eta / ext_ior),
		  k_(k / ext_ior),
		  specular_reflectance_(std::move(specular_reflectance))
	{
		if (mirror)
		{
			eta_ = Vector3(0);
			k_ = Vector3(1) / ext_ior;
		}
		auto [reflectivity, edgetint] = IorToReflectivityEdgetint(eta_, k_);
		auto F_avg = AverageFresnelConductor(reflectivity, edgetint);
		f_add_ = Sqr(F_avg) * albedo_avg_ / (Vector3(1) - F_avg * (1 - albedo_avg_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
	{
		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);
		auto normal_micro = distrib->Sample(normal, {UniformFloat(), UniformFloat()});
		DeleteDistribPointer(distrib);
		auto wi = -Reflect(-wo, normal_micro);
		if (glm::dot(wi, normal) * glm::dot(wo, normal) >= 0)
			return {Vector3(0), BsdfSamplingType::kNone};
		else
			return {wi, BsdfSamplingType::kReflection};
	}

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
	{
		if (bsdf_sampling_type == BsdfSamplingType::kNone || NotSameHemis(wo, normal))
			return Vector3(0);

		auto cos_i_n = std::fabs(glm::dot(wi, normal)),
			 cos_o_n = std::fabs(glm::dot(wo, normal));

		auto h = glm::normalize(-wi + wo);

		Vector3 F(0);
		if (mirror_)
			F = Vector3(1);
		else
			F = FresnelConductor(wi, h, eta_, k_);

		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);

		auto D = distrib->Eval(h, normal);
		auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);
		DeleteDistribPointer(distrib);

		auto weight = F * static_cast<Float>(D * G / (4 * cos_i_n * cos_o_n));
		if (specular_reflectance_ != nullptr)
			weight *= *specular_reflectance_;

		if (albedo_avg_ < kOneMinusEpsilon)
			weight += EvalMultipleScatter(cos_i_n, cos_o_n);

		return weight;
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override
	{
		if (bsdf_sampling_type == BsdfSamplingType::kNone || NotSameHemis(wo, normal))
			return 0;

		auto cos_i_n = glm::dot(wi, normal),
			 cos_o_n = glm::dot(wo, normal);

		auto h = glm::normalize(-wi + wo);

		auto distrib = InitDistrib(distrib_type_, alpha_u_, alpha_v_);
		auto D = distrib->Eval(h, normal);
		DeleteDistribPointer(distrib);

		auto jacobian = std::fabs(1 / (4 * glm::dot(wo, h)));
		return D * jacobian;
	}

private:
	bool mirror_;									//是否是镜面
	Vector3 eta_;									//材质相对折射率的实部
	Vector3 k_;										//材质相对折射率的虚部,
	std::unique_ptr<Vector3> specular_reflectance_; //镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。

	Vector3 f_add_;

	Vector3 EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
	{
		auto albedo_i = GetAlbedo(std::fabs(cos_i_n));
		auto albedo_o = GetAlbedo(std::fabs(cos_o_n));
		auto f_ms = (1 - albedo_o) * (1 - albedo_i) / (kPi * (1 - albedo_avg_));
		return f_ms * f_add_;
	}
};

NAMESPACE_END(simple_renderer)