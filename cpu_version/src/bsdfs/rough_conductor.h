#pragma once

#include "microfacet.h"
#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的导体材质派生类
class RoughConductor : public Bsdf, public Microfacet
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
	RoughConductor(const Spectrum &eta, const Spectrum &k, Float ext_ior, std::unique_ptr<Texture> specular_reflectance,
				   MicrofacetDistribType distrib_type, std::unique_ptr<Texture> alpha_u, std::unique_ptr<Texture> alpha_v)
		: Bsdf(BsdfType::kRoughConductor), Microfacet(distrib_type, std::move(alpha_u), std::move(alpha_v)),
		  eta_(eta / ext_ior), k_(k / ext_ior), specular_reflectance_(std::move(specular_reflectance))
	{
		if (Bsdf::TextureMapping())
			return;
		ComputeAlbedoTable();
		if (albedo_avg_ < 0)
			return;
		auto [reflectivity, edgetint] = IorToReflectivityEdgetint(eta_, k_);
		Spectrum F_avg = AverageFresnelConductor(reflectivity, edgetint);
		f_add_ = Sqr(F_avg) * albedo_avg_ / (Spectrum(1) - F_avg * (1 - albedo_avg_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(SamplingRecord &rec) const override
	{
		//生成光线方向
		auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);							 //景物表面沿切线方向和副切线方向的粗糙程度
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);				 //微表面分布
		auto [h, D] = distrib->Sample(rec.normal, {UniformFloat(), UniformFloat()}); //微表面法线和相应的概率（相对于宏观表面法线）
		rec.wi = -Reflect(-rec.wo, h);

		//计算光线传播概率
		Float cos_theta_i = glm::dot(-rec.wi, rec.normal); //入射光线方向和宏观表面法线方向夹角的余弦
		if (cos_theta_i < 0)
			return;
		rec.pdf = D * std::abs(1.0 / (4.0 * glm::dot(rec.wo, h)));
		if (rec.pdf < kEpsilonPdf)
			return;
		rec.type = ScatteringType::kReflect;
		
		//计算光能衰减系数
		if (!rec.get_attenuation)
			return;
		Spectrum F = FresnelConductor(rec.wi, h, eta_, k_);											  //菲涅尔项
		Float G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal), //阴影-遮蔽项
			cos_theta_o = glm::dot(rec.wo, rec.normal);												  //出射光线方向和宏观表面法线方向夹角的余弦
		rec.attenuation = F * static_cast<Float>(D * G / std::abs(4.0 * cos_theta_i * cos_theta_o));
		if (albedo_avg_ > 0)
			rec.attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
		if (specular_reflectance_)
			rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
		//因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
		rec.attenuation *= cos_theta_i;
	}

	///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
	void Eval(SamplingRecord &rec) const override
	{
		Float cos_theta_o = glm::dot(rec.wo, rec.normal); //出射光线方向和宏观表面法线方向夹角的余弦
		if (cos_theta_o < 0)
		{ //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
			//又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
			//故只需确保光线出射方向和表面法线方向在介质同侧即可
			return;
		}

		//计算光线传播概率
		auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);			 //景物表面沿切线方向和副切线方向的粗糙程度
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v); //微表面分布
		Vector3 h = glm::normalize(-rec.wi + rec.wo);				 //微表面法线
		Float D = distrib->Pdf(h, rec.normal);						 //微表面法线分布概率（相对于宏观表面法线）
		rec.pdf = D * std::abs(1.0 / (4.0 * glm::dot(rec.wo, h)));
		if (rec.pdf < kEpsilonPdf)
			return;
		rec.type = ScatteringType::kReflect;

		//计算光能衰减系数
		Spectrum F = FresnelConductor(rec.wi, h, eta_, k_);											  //菲涅尔项
		Float G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal), //阴影-遮蔽项
			cos_theta_i = glm::dot(-rec.wi, rec.normal);											  //入射光线方向和宏观表面法线方向夹角的余弦
		rec.attenuation = F * static_cast<Float>(D * G / std::abs(4.0 * cos_theta_i * cos_theta_o));
		if (albedo_avg_ > 0)
			rec.attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
		if (specular_reflectance_)
			rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
		//因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
		rec.attenuation *= cos_theta_i;
	}

	///\brief 是否映射纹理
	bool TextureMapping() const override
	{
		return Bsdf::TextureMapping() || Microfacet::TextureMapping() ||
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