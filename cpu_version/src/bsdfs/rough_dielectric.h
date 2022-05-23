#pragma once

#include "microfacet.h"
#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的电介质派生类
class RoughDielectric : public Material, public Microfacet
{
public:
	///\brief 粗糙的电介质材质
	///\param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	///\param alpha_u 沿切线（tangent）方向的粗糙度
	///\param alpha_v 沿副切线（bitangent）方向的粗糙度
	///\param int_ior 内折射率
	///\param ext_ior 外折射率
	///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	RoughDielectric(Float int_ior, Float ext_ior,
					std::unique_ptr<Texture> specular_reflectance,
					std::unique_ptr<Texture> specular_transmittance,
					MicrofacetDistribType distrib_type,
					std::unique_ptr<Texture> alpha_u, std::unique_ptr<Texture> alpha_v)
		: Material(MaterialType::kRoughDielectric),
		  Microfacet(distrib_type, std::move(alpha_u), std::move(alpha_v)),
		  eta_(int_ior / ext_ior), eta_inv_(ext_ior / int_ior),
		  specular_reflectance_(std::move(specular_reflectance)),
		  specular_transmittance_(std::move(specular_transmittance)),
		  f_add_(0), f_add_inv_(0), ratio_t_(0), ratio_t_inv_(0)
	{
		if (Material::TextureMapping())
			return;
		ComputeAlbedoTable();
		if (albedo_avg_ < 0)
			return;
		Float F_avg = AverageFresnel(eta_), F_avg_inv = AverageFresnel(eta_inv_);
		f_add_ = F_avg * albedo_avg_ / (1.0 - F_avg * (1.0 - albedo_avg_));
		f_add_inv_ = F_avg_inv * albedo_avg_ / (1.0 - F_avg_inv * (1.0 - albedo_avg_));
		ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * Sqr(eta_) /
				   ((1.0 - F_avg) + (1.0 - F_avg_inv) * Sqr(eta_));
		ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * Sqr(eta_inv_) /
					   ((1.0 - F_avg_inv) + (1.0 - F_avg) * Sqr(eta_inv_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(BsdfSampling &bs) const override
	{
		Float eta = bs.inside ? eta_inv_ : eta_,   //相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
			eta_inv = bs.inside ? eta_ : eta_inv_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
			ratio_t = bs.inside ? ratio_t_inv_ : ratio_t_,
			  ratio_t_inv = bs.inside ? ratio_t_ : ratio_t_inv_;
		auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		// Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
		distrib->ScaleAlpha(1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-bs.wo, bs.normal))));
		auto [h, D] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
		if (D < kEpsilonPdf)
			return;
		Float F = Fresnel(-bs.wo, h, eta_inv);
		if (UniformFloat() < F)
		{ //抽样反射光线
			bs.wi = -Reflect(-bs.wo, h);
			Float cos_theta_i = glm::dot(bs.wi, bs.normal);
			if (cos_theta_i >= 0)
				return;
			bs.pdf = F * D * std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
			if (bs.pdf < kEpsilonPdf || !bs.get_attenuation)
				return;
			Float G = distrib->SmithG1(-bs.wi, h, bs.normal) * distrib->SmithG1(bs.wo, h, bs.normal),
				  cos_theta_o = glm::dot(bs.wo, bs.normal);
			bs.attenuation = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
			if (albedo_avg_ > 0)
			{
				Float weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, bs.inside);
				bs.attenuation += Spectrum(weight_loss);
			}
			if (specular_reflectance_)
				bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
		}
		else
		{ //抽样折射光线
			bs.wi = -Refract(-bs.wo, h, eta_inv);
			Float cos_theta_i = glm::dot(bs.wi, bs.normal);
			if (cos_theta_i <= 0)
				return;
			bs.normal = -bs.normal;
			bs.inside = !bs.inside;
			h = -h;
			eta_inv = eta;
			ratio_t = ratio_t_inv;
			F = Fresnel(bs.wi, h, eta_inv);
			Float cos_i_h = glm::dot(-bs.wi, h),
				  cos_o_h = glm::dot(bs.wo, h);
			bs.pdf = (1.0 - F) * D * std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));
			if (bs.pdf < kEpsilonPdf || !bs.get_attenuation)
				return;
			Float G = distrib->SmithG1(-bs.wi, h, bs.normal) * distrib->SmithG1(bs.wo, h, bs.normal),
				  cos_theta_o = glm::dot(bs.wo, bs.normal);
			bs.attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
											   (cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (albedo_avg_ > 0)
			{
				Float weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, bs.inside);
				bs.attenuation += Spectrum(weight_loss);
			}
			if (specular_transmittance_)
				bs.attenuation *= specular_transmittance_->Color(bs.texcoord);
			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			bs.attenuation *= Sqr(eta_inv);
		}
	}

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
				  bool inside) const override
	{
		Float eta_inv = inside ? eta_ : eta_inv_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
			ratio_t = inside ? ratio_t_inv_ : ratio_t_;

		Float cos_theta_o = glm::dot(wo, normal),
			  cos_theta_i = glm::dot(-wi, normal),
			  F = 0;
		auto h = Vector3(0);
		bool relfect = cos_theta_o > 0;
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
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		Float D = distrib->Pdf(h, normal),
			  G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);
		if (relfect)
		{
			auto albedo = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
			if (albedo_avg_ > 0)
			{
				Float weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, inside);
				albedo += Spectrum(weight_loss);
			}
			if (specular_reflectance_)
				albedo *= specular_reflectance_->Color(texcoord);
			return albedo;
		}
		else
		{
			Float cos_i_h = glm::dot(-wi, h), cos_o_h = glm::dot(wo, h);
			auto attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
												 (cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (albedo_avg_ > 0)
			{
				auto weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, inside);
				attenuation += Spectrum(weight_loss);
			}
			if (specular_transmittance_)
				attenuation *= specular_transmittance_->Color(texcoord);
			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			attenuation *= Sqr(eta_inv);
			return attenuation;
		}
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
			  bool inside) const override
	{
		Float eta_inv = inside ? eta_ : eta_inv_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
			cos_theta_o = glm::dot(wo, normal);
		bool relfect = cos_theta_o > 0;
		auto h = Vector3(0);
		if (relfect)
			h = glm::normalize(-wi + wo);
		else
		{
			h = glm::normalize(-eta_inv * wi + wo);
			if (NotSameHemis(h, normal))
				h = -h;
		}
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		Float D = distrib->Pdf(h, normal),
			  F = Fresnel(wi, h, eta_inv);
		if (D < kEpsilon)
			return 0;
		if (relfect)
			return F * D * std::abs(1.0 / (4.0 * glm::dot(wo, h)));
		else
			return (1.0 - F) * D * std::abs(glm::dot(wo, h) / Sqr(eta_inv * glm::dot(-wi, h) + glm::dot(wo, h)));
	}

	///\brief 是否映射纹理
	bool TextureMapping() const override
	{
		return Material::TextureMapping() ||
               Microfacet::TextureMapping() ||
			   specular_reflectance_ && !specular_reflectance_->Constant() ||
			   specular_transmittance_ && !specular_transmittance_->Constant();
	}

private:
	///\brief 补偿多次散射后又射出的光能
	Float EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o, bool inside) const
	{
		Float f_add = inside ? f_add_inv_ : f_add_,
			  albedo_i = GetAlbedo(std::abs(cos_theta_i)),
			  albedo_o = GetAlbedo(std::abs(cos_theta_o)),
			  f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
		return f_ms * f_add;
	}

	Float eta_;										  //光线射入材质的相对折射率
	Float eta_inv_;									  //光线射出材质的相对折射率
	std::unique_ptr<Texture> specular_reflectance_;	  // 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	std::unique_ptr<Texture> specular_transmittance_; // 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	Float f_add_;									  //入射光线在物体外部，补偿多次散射后出射光能的系数
	Float f_add_inv_;								  //入射光线在物体内部，补偿多次散射后出射光能的系数
	Float ratio_t_;									  //入射光线在物体外部，补偿多次散射后出射光能中折射的比例
	Float ratio_t_inv_;								  //入射光线在物体内部，补偿多次散射后出射光能中折射的比例
};

NAMESPACE_END(raytracer)