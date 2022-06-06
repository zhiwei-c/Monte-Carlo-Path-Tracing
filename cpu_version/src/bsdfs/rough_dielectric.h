#pragma once

#include "microfacet.h"
#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 粗糙的电介质派生类
class RoughDielectric : public Bsdf, public Microfacet
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
	RoughDielectric(Float int_ior, Float ext_ior, std::unique_ptr<Texture> specular_reflectance,
					std::unique_ptr<Texture> specular_transmittance, MicrofacetDistribType distrib_type,
					std::unique_ptr<Texture> alpha_u, std::unique_ptr<Texture> alpha_v)
		: Bsdf(BsdfType::kRoughDielectric), Microfacet(distrib_type, std::move(alpha_u), std::move(alpha_v)),
		  eta_(int_ior / ext_ior), eta_inv_(ext_ior / int_ior), specular_reflectance_(std::move(specular_reflectance)),
		  specular_transmittance_(std::move(specular_transmittance)), f_add_(0), f_add_inv_(0), ratio_t_(0), ratio_t_inv_(0)
	{
		if (Bsdf::TextureMapping())
			return;
		ComputeAlbedoTable();
		if (albedo_avg_ < 0)
			return;
		Float F_avg = AverageFresnel(eta_),
			  F_avg_inv = AverageFresnel(eta_inv_);
		f_add_ = F_avg * albedo_avg_ / (1.0 - F_avg * (1.0 - albedo_avg_));
		ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * Sqr(eta_) / ((1.0 - F_avg) + (1.0 - F_avg_inv) * Sqr(eta_));
		f_add_inv_ = F_avg_inv * albedo_avg_ / (1.0 - F_avg_inv * (1.0 - albedo_avg_));
		ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * Sqr(eta_inv_) / ((1.0 - F_avg_inv) + (1.0 - F_avg) * Sqr(eta_inv_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(SamplingRecord &rec) const override
	{
		Float eta = rec.inside ? eta_inv_ : eta_,				//相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
			eta_inv = rec.inside ? eta_ : eta_inv_,				//相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
			ratio_t = rec.inside ? ratio_t_inv_ : ratio_t_,		//补偿多次散射后出射光能中折射的比例
			ratio_t_inv = rec.inside ? ratio_t_ : ratio_t_inv_; //光线逆向传播时，补偿多次散射后出射光能中折射的比例

		//抽样微表面法线
		auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);			 //景物表面沿切线方向和副切线方向的粗糙程度
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v); //微表面分布
		// Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
		distrib->ScaleAlpha(1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-rec.wo, rec.normal))));
		auto [h, D] = distrib->Sample(rec.normal, {UniformFloat(), UniformFloat()}); //微表面法线和相应的概率（相对于宏观表面法线）
		if (D < kEpsilonPdf)
			return;

		Float F = Fresnel(-rec.wo, h, eta_inv), //菲涅尔项
			cos_theta_i = 0;					//入射光线方向和宏观表面法线方向夹角的余弦
		if (UniformFloat() < F)
		{ //抽样反射光线
			//生成光线方向
			rec.wi = -Reflect(-rec.wo, h);
			 cos_theta_i = glm::dot(-rec.wi, rec.normal); 
			if (cos_theta_i < kEpsilon)
				return;

			//计算光线传播概率
			rec.pdf = F * D * std::abs(1.0 / (4.0 * glm::dot(rec.wo, h)));
			if (rec.pdf < kEpsilonPdf)
				return;
			rec.type = ScatteringType::kReflect;

			//计算光能衰减系数
			if (!rec.get_attenuation)
				return;
			Float G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal), //阴影-遮蔽项
				cos_theta_o = glm::dot(rec.wo, rec.normal);												  //出射光线方向和宏观表面法线方向夹角的余弦
			rec.attenuation = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
			if (albedo_avg_ > 0)
			{ //补偿多次散射后又射出的光能
				Float weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
				rec.attenuation += Spectrum(weight_loss);
			}
			if (specular_reflectance_)
				rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
		}
		else
		{ //抽样折射光线
			//生成光线方向
			rec.wi = -Refract(-rec.wo, h, eta_inv);
			cos_theta_i = glm::dot(rec.wi, rec.normal); //入射光线方向和宏观表面法线方向夹角的余弦
			if (cos_theta_i < kEpsilon)
				return;
			{ //光线折射时穿过了介质，为了使得光线入射方向和表面法线方向夹角的余弦仍小于零，需做一些相应处理
				rec.normal = -rec.normal;
				rec.inside = !rec.inside;
				eta_inv = eta;
				h = -h;
				ratio_t = ratio_t_inv;
			}

			//计算光线传播概率
			F = Fresnel(rec.wi, h, eta_inv);
			Float cos_i_h = glm::dot(-rec.wi, h), //入射光线方向和微表面法线方向夹角的余弦
				cos_o_h = glm::dot(rec.wo, h);	  //出射光线方向和微表面法线方向夹角的余弦
			rec.pdf = (1.0 - F) * D * std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));
			if (rec.pdf < kEpsilonPdf)
				return;
			rec.type = ScatteringType::kTransimission;

			//计算光能衰减系数
			if (!rec.get_attenuation)
				return;
			Float G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal), //阴影-遮蔽项
				cos_theta_o = glm::dot(rec.wo, rec.normal);												  //出射光线方向和宏观表面法线方向夹角的余弦
			rec.attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
												(cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (albedo_avg_ > 0)
			{ //补偿多次散射后又射出的光能
				Float weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
				rec.attenuation += Spectrum(weight_loss);
			}
			if (specular_transmittance_)
				rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			rec.attenuation *= Sqr(eta_inv);
		}
		//因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
		rec.attenuation *= cos_theta_i;
	}

	///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
	void Eval(SamplingRecord &rec) const override
	{
		//计算微表面法线方向
		Float eta_inv = rec.inside ? eta_ : eta_inv_,	//相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
			cos_theta_o = glm::dot(rec.wo, rec.normal); //出射光线方向和宏观表面法线方向夹角的余弦
		auto h = Vector3(0);							//微表面法线
		bool relfect = cos_theta_o > 0;					//出射光线是否是反射光线
		if (relfect)
			h = glm::normalize(-rec.wi + rec.wo);
		else
		{
			h = glm::normalize(-eta_inv * rec.wi + rec.wo);
			if (NotSameHemis(h, rec.normal))
				h = -h;
		}
		//计算光线传播概率
		auto [alpha_u, alpha_v] = GetAlpha(rec.texcoord);			 //景物表面沿切线方向和副切线方向的粗糙程度
		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v); //微表面分布
		Float D = distrib->Pdf(h, rec.normal),						 //微表面法线分布概率（相对于宏观表面法线）
			F = Fresnel(rec.wi, h, eta_inv),						 //菲涅尔项
			cos_i_h = glm::dot(-rec.wi, h),							 //入射光线方向和微表面法线方向夹角的余弦
			cos_o_h = glm::dot(rec.wo, h);							 //出射光线方向和微表面法线方向夹角的余弦
		rec.pdf = relfect ? F * D * std::abs(1.0 / (4.0 * cos_o_h))
						  : (1.0 - F) * D * std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));
		if (rec.pdf < kEpsilonPdf)
			return;
		rec.type = relfect ? ScatteringType::kReflect : ScatteringType::kTransimission;
		//计算光能衰减系数
		Float ratio_t = rec.inside ? ratio_t_inv_ : ratio_t_,										//补偿多次散射后出射光能中折射的比例
			cos_theta_i = glm::dot(-rec.wi, rec.normal),											//入射光线方向和宏观表面法线方向夹角的余弦
			G = distrib->SmithG1(-rec.wi, h, rec.normal) * distrib->SmithG1(rec.wo, h, rec.normal); //阴影-遮蔽项
		if (relfect)
		{
			rec.attenuation = Spectrum(F * D * G / (4.0 * std::abs(cos_theta_i * cos_theta_o)));
			if (albedo_avg_ > 0)
			{
				Float weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
				rec.attenuation += Spectrum(weight_loss);
			}
			if (specular_reflectance_)
				rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
		}
		else
		{
			rec.attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
												(cos_theta_i * cos_theta_o * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (albedo_avg_ > 0)
			{
				Float weight_loss = ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
				rec.attenuation += Spectrum(weight_loss);
			}
			if (specular_transmittance_)
				rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			rec.attenuation *= Sqr(eta_inv);
		}
		//因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
		rec.attenuation *= cos_theta_i;
	}

	///\brief 是否映射纹理
	bool TextureMapping() const override
	{
		return Bsdf::TextureMapping() || Microfacet::TextureMapping() ||
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

	Float f_add_;									  //入射光线在物体外部时，补偿多次散射后出射光能的系数
	Float f_add_inv_;								  //入射光线在物体内部时，补偿多次散射后出射光能的系数
	Float ratio_t_;									  //入射光线在物体外部时，补偿多次散射后出射光能中折射的比例
	Float ratio_t_inv_;								  //入射光线在物体内部时，补偿多次散射后出射光能中折射的比例
	Float eta_;										  //光线射入材质的相对折射率
	Float eta_inv_;									  //光线射出材质的相对折射率
	std::unique_ptr<Texture> specular_reflectance_;	  // 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	std::unique_ptr<Texture> specular_transmittance_; // 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)