#pragma once

#include "microfacet.h"

NAMESPACE_BEGIN(simple_renderer)

class RoughDielectric : public Microfacet
{
public:
	/**
	 * \brief 粗糙的电介质材质
	 * \param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	 * \param alpha_u 沿切线（tangent）方向的粗糙度
	 * \param alpha_v 沿副切线（bitangent）方向的粗糙度
	 * \param int_ior 内折射率
	 * \param ext_ior 外折射率
	 * \param specular_reflectance 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
	 * \param specular_transmittance 镜面透射系数。注意，对于物理真实感绘制，应默认为 1。
	 */
	RoughDielectric(MicrofacetDistribType distrib_type,
					std::unique_ptr<Texture> alpha_u,
					std::unique_ptr<Texture> alpha_v,
					Float int_ior,
					Float ext_ior,
					std::unique_ptr<Texture> specular_reflectance = nullptr,
					std::unique_ptr<Texture> specular_transmittance = nullptr)
		: Microfacet(MaterialType::kRoughDielectric,
					 distrib_type,
					 std::move(alpha_u),
					 std::move(alpha_v)),
		  eta_(int_ior / ext_ior),
		  eta_inv_(ext_ior / int_ior),
		  specular_reflectance_(std::move(specular_reflectance)),
		  specular_transmittance_(std::move(specular_transmittance))
	{
		f_add_ = 0,
		f_add_inv_ = 0,
		ratio_t_ = 0,
		ratio_t_inv_ = 0;

		if (Microfacet::TextureMapping())
			return;

		auto F_avg = AverageFresnelDielectric(eta_);
		f_add_ = F_avg * albedo_avg_ / (1 - F_avg * (1 - albedo_avg_));

		auto F_avg_inv = AverageFresnelDielectric(eta_inv_);
		f_add_inv_ = F_avg_inv * albedo_avg_ / (1 - F_avg_inv * (1 - albedo_avg_));

		ratio_t_ = (1 - F_avg) * (1 - F_avg_inv) * Sqr(eta_) /
				   ((1 - F_avg) + (1 - F_avg_inv) * Sqr(eta_));

		ratio_t_inv_ = (1 - F_avg_inv) * (1 - F_avg) * Sqr(eta_inv_) /
					   ((1 - F_avg_inv) + (1 - F_avg) * Sqr(eta_inv_));
	}

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(BsdfSampling &bs) const override
	{
		auto eta = bs.inside ? eta_inv_ : eta_;		//相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
		auto eta_inv = bs.inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

		auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);

		// Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
		distrib->ScaleAlpha(1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-bs.wo, bs.normal))));

		auto [normal_micro, D] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});

		if (D < kEpsilon)
			return;

		auto F = Fresnel(-bs.wo, normal_micro, eta_inv);
		auto sample_x = UniformFloat();
		if (sample_x < F)
		{
			bs.wi = -Reflect(-bs.wo, normal_micro);
			if (glm::dot(bs.wi, bs.normal) >= 0)
				return;
		}
		else
		{
			bs.wi = -Refract(-bs.wo, normal_micro, eta_inv);
			if (glm::dot(bs.wi, bs.normal) <= 0)
				return;
			bs.normal = -bs.normal;
			bs.inside = !bs.inside;
		}

		bs.pdf = Pdf(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
		if (bs.pdf < kEpsilonPdf)
		{
			bs.pdf = 0;
			return;
		}
		if (!Microfacet::TextureMapping() && alpha_u > 0.01 && alpha_v > 0.01 && bs.pdf < kEpsilonL)
		{
			bs.pdf = 0;
			return;
		}

		if (bs.get_weight)
			bs.weight = Eval(bs.wi, bs.wo, bs.normal, bs.texcoord, bs.inside);
	}

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
	{
		auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
		auto ratio_t = inside ? ratio_t_inv_ : ratio_t_;
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);

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

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		auto D = distrib->Pdf(h, normal);
		auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);

		if (relfect)
		{
			auto albedo = Spectrum(F * D * G / (4 * std::fabs(cos_i_n * cos_o_n)));
			if (specular_reflectance_)
			{
				if (texcoord != nullptr)
					albedo *= specular_reflectance_->GetPixel(*texcoord);
				else
					albedo *= specular_reflectance_->GetPixel(Vector2(0));
			}
			if (!Microfacet::TextureMapping() && albedo_avg_ < kOneMinusEpsilon)
			{
				auto weight_loss = (1 - ratio_t) * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
				albedo += Spectrum(weight_loss);
			}

			return albedo;
		}
		else
		{
			auto cos_i_h = glm::dot(-wi, h),
				 cos_o_h = glm::dot(wo, h);
			auto weight = Spectrum(std::fabs(cos_i_h * cos_o_h * (1 - F) * G * D /
											 (cos_i_n * cos_o_n * Sqr(eta_inv * cos_i_h + cos_o_h))));
			if (specular_transmittance_)
			{
				if (texcoord != nullptr)
					weight *= specular_transmittance_->GetPixel(*texcoord);
				else
					weight *= specular_transmittance_->GetPixel(Vector2(0));
			}
			if (!Microfacet::TextureMapping() && albedo_avg_ < kOneMinusEpsilon)
			{
				auto weight_loss = ratio_t * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
				weight += Spectrum(weight_loss);
			}

			//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
			weight *= Sqr(eta_inv);

			return weight;
		}
	}

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
	{
		auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
		auto [alpha_u, alpha_v] = GetAlpha(texcoord);

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

		auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
		auto D = distrib->Pdf(h, normal);

		if (D < kEpsilon)
			return 0;

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

	///\brief 是否映射纹理
	bool TextureMapping() const override { return Microfacet::TextureMapping() ||
												  (specular_reflectance_ && !specular_reflectance_->Constant()) ||
												  (specular_transmittance_ && !specular_transmittance_->Constant()); }

private:
	Float eta_;										  //光线射入材质的相对折射率
	Float eta_inv_;									  //光线射出材质的相对折射率
	Float f_add_;									  //入射光线在物体内部，补偿多次散射后出射光能的系数
	Float f_add_inv_;								  //入射光线在物体外部，补偿多次散射后出射光能的系数
	Float ratio_t_;									  //入射光线在物体内部，补偿多次散射后出射光能，折射的比例
	Float ratio_t_inv_;								  //入射光线在物体外部，补偿多次散射后出射光能，折射的比例
	std::unique_ptr<Texture> specular_reflectance_;	  // 镜面反射系数。注意，对于物理真实感绘制，应默认为 1。
	std::unique_ptr<Texture> specular_transmittance_; // 镜面透射系数。注意，对于物理真实感绘制，应默认为 1。

	///\brief 补偿多次散射后又射出的光能
	Float EvalMultipleScatter(Float cos_i_n, Float cos_o_n, bool inside) const
	{
		auto f_add = inside ? f_add_inv_ : f_add_;
		auto albedo_i = GetAlbedo(std::fabs(cos_i_n));
		auto albedo_o = GetAlbedo(std::fabs(cos_o_n));
		auto f_ms = (1 - albedo_o) * (1 - albedo_i) / (kPi * (1 - albedo_avg_));
		return f_ms * f_add;
	}
};

NAMESPACE_END(simple_renderer)