#pragma once

#include "microfacet.h"
#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的导体材质派生类
class RoughConductor : public Material, public Microfacet
{
public:
	///\brief 粗糙的导体材质
	///\param eta 材质折射率的实部
	///\param k 材质折射率的虚部（消光系数）
	///\param ext_ior 外折射率
	///\param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1
	///\param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	///\param alpha_u 沿切线（tangent）方向的粗糙度
	///\param alpha_v 沿副切线（bitangent）方向的粗糙度
	RoughConductor(const Spectrum &eta, const Spectrum &k, Float ext_ior,
				   std::unique_ptr<Texture> specular_reflectance, MicrofacetDistribType distrib_type,
				   std::unique_ptr<Texture> alpha_u, std::unique_ptr<Texture> alpha_v)
		: Material(MaterialType::kRoughConductor),
		  Microfacet(distrib_type, std::move(alpha_u), std::move(alpha_v)),
		  eta_(eta / ext_ior), k_(k / ext_ior),
		  specular_reflectance_(std::move(specular_reflectance))
	{
		if (Material::TextureMapping())
			return;
		ComputeAlbedoTable();
		if (albedo_avg_ < 0)
			return;
		auto reflectivity = Spectrum(0), edgetint = Spectrum(0);
		IorToReflectivityEdgetint(eta_, k_, reflectivity, edgetint);
		auto F_avg = AverageFresnelConductor(reflectivity, edgetint);
		f_add_ = Sqr(F_avg) * albedo_avg_ / (Spectrum(1) - F_avg * (1 - albedo_avg_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(BsdfSampling &bs) const override
	{
		auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		auto [h, D] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
		if (D < kEpsilonPdf)
			return;
		bs.wi = -Reflect(-bs.wo, h);
		Float cos_theta_i = glm::dot(bs.wi, bs.normal);
		if (cos_theta_i >= 0)
			return;
		bs.pdf = D * std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
		if (bs.pdf < kEpsilonPdf || !bs.get_attenuation)
            return;
		Spectrum F = FresnelConductor(bs.wi, h, eta_, k_);
		Float G = distrib->SmithG1(-bs.wi, h, bs.normal) * distrib->SmithG1(bs.wo, h, bs.normal),
			  cos_theta_o = glm::dot(bs.wo, bs.normal);
		bs.attenuation = F * static_cast<Float>(D * G / std::abs(4.0 * cos_theta_i * cos_theta_o));
		if (specular_reflectance_)
			bs.attenuation *= specular_reflectance_->Color(bs.texcoord);
		if (albedo_avg_ > 0)
			bs.attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
	}

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
				  bool inside) const override
	{
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		Vector3 h = glm::normalize(-wi + wo);
		Float D = distrib->Pdf(h, normal),
			  G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal),
			  cos_theta_i = glm::dot(wi, normal),
			  cos_theta_o = glm::dot(wo, normal);
		Spectrum F = FresnelConductor(wi, h, eta_, k_),
				 albedo = F * static_cast<Float>(D * G / std::abs(4.0 * cos_theta_i * cos_theta_o));
		if (specular_reflectance_)
			albedo *= specular_reflectance_->Color(texcoord);
		if (albedo_avg_ > 0)
			albedo += EvalMultipleScatter(cos_theta_i, cos_theta_o);
		return albedo;
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
			  bool inside) const override
	{
		// 表面法线方向，光线入射和出射需在介质同侧
		if (NotSameHemis(wo, normal))
			return 0;
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		Vector3 h = glm::normalize(-wi + wo);
		Float D = distrib->Pdf(h, normal);
		if (D < kEpsilonPdf)
			return 0;
		return D * std::abs(1.0 / (4.0 * glm::dot(wo, h)));
	}

	///\brief 是否映射纹理
	bool TextureMapping() const override
	{
		return Material::TextureMapping() ||
               Microfacet::TextureMapping() ||
			   specular_reflectance_ && !specular_reflectance_->Constant();
	}

private:
	///\brief 补偿多次散射后又射出的光能
	Spectrum EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o) const
	{
		Float albedo_i = GetAlbedo(std::abs(cos_theta_i)),
			  albedo_o = GetAlbedo(std::abs(cos_theta_o)),
			  f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
		return f_ms * f_add_;
	}

	Spectrum eta_;									//材质相对折射率的实部
	Spectrum k_;									//材质相对折射率的虚部（消光系数）
	std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	Spectrum f_add_;								//补偿多次散射后出射光能的系数
};

NAMESPACE_END(raytracer)