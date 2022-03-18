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
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha_u 沿切线（tangent）方向的粗糙度
	 * \param alpha_v 沿副切线（bitangent）方向的粗糙度
	 * \param mirror 是否是镜面（全反射）
	 * \param eta 材质折射率的实部
	 * \param k 材质折射率的虚部（消光系数）
	 * \param ext_ior 外折射率
	 * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1
	 */
	RoughConductor(const std::string &id,
				   MicrofacetDistribType distrib_type,
				   std::unique_ptr<Texture> alpha_u,
				   std::unique_ptr<Texture> alpha_v,
				   bool mirror,
				   const Spectrum &eta,
				   const Spectrum &k,
				   Float ext_ior,
				   std::unique_ptr<Texture> specular_reflectance = nullptr)
		: Microfacet(id,
					 MaterialType::kRoughConductor,
					 distrib_type,
					 std::move(alpha_u),
					 std::move(alpha_v)),
		  mirror_(mirror),
		  eta_(eta / ext_ior),
		  k_(k / ext_ior),
		  specular_reflectance_(std::move(specular_reflectance))
	{
		if (mirror)
		{
			eta_ = Spectrum(0);
			k_ = Spectrum(1) / ext_ior;
		}

		f_add_ = Spectrum(0);
		if (Microfacet::TextureMapping())
			return;

		auto [reflectivity, edgetint] = IorToReflectivityEdgetint(eta_, k_);
		auto F_avg = AverageFresnelConductor(reflectivity, edgetint);
		f_add_ = Sqr(F_avg) * albedo_avg_ / (Spectrum(1) - F_avg * (1 - albedo_avg_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
	{
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		auto [normal_micro, pdf] = distrib->Sample(normal, {UniformFloat(), UniformFloat()});

		if (pdf < kEpsilon)
			return BsdfSampling();

		BsdfSampling bs;

		bs.wi = -Reflect(-wo, normal_micro);
		if (glm::dot(bs.wi, normal) >= 0)
			return BsdfSampling();

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

		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto cos_i_n = std::fabs(glm::dot(wi, normal)),
			 cos_o_n = std::fabs(glm::dot(wo, normal));

		auto h = glm::normalize(-wi + wo);

		Spectrum F(0);
		if (mirror_)
			F = Spectrum(1);
		else
			F = FresnelConductor(wi, h, eta_, k_);

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);

		auto D = distrib->Pdf(h, normal);
		auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);

		auto albedo = F * static_cast<Float>(D * G / (4 * cos_i_n * cos_o_n));

		if (specular_reflectance_)
		{
			if (texcoord != nullptr)
				albedo *= specular_reflectance_->GetPixel(*texcoord);
			else
				albedo *= specular_reflectance_->GetPixel(Vector2(0));
		}

		if (!Microfacet::TextureMapping() && albedo_avg_ < kOneMinusEpsilon)
			albedo += EvalMultipleScatter(cos_i_n, cos_o_n);

		return albedo;
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
	{
		if (NotSameHemis(wo, normal))
			return 0;

		if (glm::dot(wi, normal) * glm::dot(wo, normal) >= 0)
			return 0;

		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto h = glm::normalize(-wi + wo);

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		auto D = distrib->Pdf(h, normal);
		if (D < kEpsilon)
			return 0;

		auto jacobian = std::fabs(1 / (4 * glm::dot(wo, h)));
		return D * jacobian;
	}

	bool TextureMapping() const override { return Microfacet::TextureMapping() || (specular_reflectance_ && !specular_reflectance_->Constant()); }

private:
	bool mirror_;  //是否是镜面
	Spectrum eta_; //材质相对折射率的实部
	Spectrum k_;   //材质相对折射率的虚部,
	Spectrum f_add_;
	std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数。注意，对于物理真实感绘制，不应设置此参数。

	Spectrum EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
	{
		auto albedo_i = GetAlbedo(std::fabs(cos_i_n));
		auto albedo_o = GetAlbedo(std::fabs(cos_o_n));
		auto f_ms = (1 - albedo_o) * (1 - albedo_i) / (kPi * (1 - albedo_avg_));
		return f_ms * f_add_;
	}
};

NAMESPACE_END(simple_renderer)