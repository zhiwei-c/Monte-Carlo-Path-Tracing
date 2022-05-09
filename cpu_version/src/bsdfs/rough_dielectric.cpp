#include "rough_dielectric.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 粗糙的电介质材质
RoughDielectric::RoughDielectric(Float int_ior,
								 Float ext_ior,
								 std::unique_ptr<Texture> specular_reflectance,
								 std::unique_ptr<Texture> specular_transmittance,
								 MicrofacetDistribType distrib_type,
								 std::unique_ptr<Texture> alpha_u,
								 std::unique_ptr<Texture> alpha_v)
	: Material(MaterialType::kRoughDielectric),
	  Microfacet(distrib_type,
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

	if (Material::TextureMapping())
	{
		albedo_avg_ = -1;
		return;
	}

	ComputeAlbedoTable();
	if (albedo_avg_ < 0)
		return;

	auto F_avg = AverageFresnel(eta_);
	f_add_ = F_avg * albedo_avg_ / (1.0 - F_avg * (1.0 - albedo_avg_));

	auto F_avg_inv = AverageFresnel(eta_inv_);
	f_add_inv_ = F_avg_inv * albedo_avg_ / (1 - F_avg_inv * (1 - albedo_avg_));

	ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * Sqr(eta_) /
			   ((1.0 - F_avg) + (1.0 - F_avg_inv) * Sqr(eta_));

	ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * Sqr(eta_inv_) /
				   ((1.0 - F_avg_inv) + (1.0 - F_avg) * Sqr(eta_inv_));
}

///\brief 根据光线出射方向和表面法线方向抽样光线入射方向，法线方向已被处理至与光线出射方向夹角大于90度
void RoughDielectric::Sample(BsdfSampling &bs) const
{
	auto eta = bs.inside ? eta_inv_ : eta_;		//相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
	auto eta_inv = bs.inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
	auto ratio_t = bs.inside ? ratio_t_inv_ : ratio_t_;
	auto ratio_t_inv = bs.inside ? ratio_t_ : ratio_t_inv_;

	auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);

	auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);

	// Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
	distrib->ScaleAlpha(1.2 - 0.2 * std::sqrt(std::fabs(glm::dot(-bs.wo, bs.normal))));

	auto [h, D] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});
	if (D < kEpsilon)
		return;

	auto F = Fresnel(-bs.wo, h, eta_inv);
	if (UniformFloat() < F)
	{
		bs.wi = -Reflect(-bs.wo, h);
		auto cos_i_n = glm::dot(bs.wi, bs.normal);
		if (cos_i_n >= 0)
			return;
		auto jacobian = std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
		bs.pdf = F * D * jacobian;
		if (bs.pdf < kEpsilonPdf || albedo_avg_ > -1 && bs.pdf < kEpsilonL)
		{
			bs.pdf = 0;
			return;
		}

		if (!bs.get_attenuation)
			return;
		auto G = distrib->SmithG1(-bs.wi, h, bs.normal) *
				 distrib->SmithG1(bs.wo, h, bs.normal);
		auto cos_o_n = glm::dot(bs.wo, bs.normal);
		auto albedo = Spectrum(F * D * G / (4.0 * std::abs(cos_i_n * cos_o_n)));
		if (specular_reflectance_)
			albedo *= specular_reflectance_->Color(bs.texcoord);
		if (albedo_avg_ > 0)
		{
			auto weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_i_n, cos_o_n, bs.inside);
			albedo += Spectrum(weight_loss);
		}
		bs.attenuation = albedo;
	}
	else
	{
		bs.wi = -Refract(-bs.wo, h, eta_inv);
		auto cos_i_n = glm::dot(bs.wi, bs.normal);
		if (cos_i_n <= 0)
			return;

		bs.normal = -bs.normal;
		bs.inside = !bs.inside;
		h = -h;
		eta_inv = eta;
		ratio_t = ratio_t_inv;

		F = Fresnel(bs.wi, h, eta_inv);
		auto cos_i_h = glm::dot(-bs.wi, h),
			 cos_o_h = glm::dot(bs.wo, h);
		auto jacobian = std::abs(cos_o_h / Sqr(eta_inv * cos_i_h + cos_o_h));

		bs.pdf = (1.0 - F) * D * jacobian;
		if (bs.pdf < kEpsilonPdf || albedo_avg_ > -1 && bs.pdf < kEpsilonL)
		{
			bs.pdf = 0;
			return;
		}

		if (!bs.get_attenuation)
			return;
		auto G = distrib->SmithG1(-bs.wi, h, bs.normal) *
				 distrib->SmithG1(bs.wo, h, bs.normal);
		auto cos_o_n = glm::dot(bs.wo, bs.normal);
		auto attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
											 (cos_i_n * cos_o_n * Sqr(eta_inv * cos_i_h + cos_o_h))));

		if (specular_transmittance_)
			attenuation *= specular_transmittance_->Color(bs.texcoord);

		if (albedo_avg_ > 0)
		{
			auto weight_loss = ratio_t * EvalMultipleScatter(cos_i_n, cos_o_n, bs.inside);
			attenuation += Spectrum(weight_loss);
		}

		//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
		attenuation *= Sqr(eta_inv);
		bs.attenuation = attenuation;
	}
}

///\brief 根据光线入射方向、出射方向和表面法线方向，计算 BSDF 权重，法线方向已被处理至与光线入射方向夹角大于90度
Spectrum RoughDielectric::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
	auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
	auto ratio_t = inside ? ratio_t_inv_ : ratio_t_;
	auto [alpha_u, alpha_v] = GetAlpha(texcoord);

	auto cos_o_n = glm::dot(wo, normal);
	auto cos_i_n = glm::dot(-wi, normal);

	auto h = Vector3(0);
	auto F = static_cast<Float>(0);
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
		auto albedo = Spectrum(F * D * G / (4.0 * std::abs(cos_i_n * cos_o_n)));
		if (specular_reflectance_)
			albedo *= specular_reflectance_->Color(texcoord);

		if (albedo_avg_ > 0)
		{
			auto weight_loss = (1.0 - ratio_t) * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
			albedo += Spectrum(weight_loss);
		}

		return albedo;
	}
	else
	{
		auto cos_i_h = glm::dot(-wi, h),
			 cos_o_h = glm::dot(wo, h);
		auto attenuation = Spectrum(std::abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
											 (cos_i_n * cos_o_n * Sqr(eta_inv * cos_i_h + cos_o_h))));
		if (specular_transmittance_)
			attenuation *= specular_transmittance_->Color(texcoord);

		if (albedo_avg_ > 0)
		{
			auto weight_loss = ratio_t * EvalMultipleScatter(cos_i_n, cos_o_n, inside);
			attenuation += Spectrum(weight_loss);
		}

		//光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
		attenuation *= Sqr(eta_inv);

		return attenuation;
	}
}

///\brief 根据光线入射方向和表面法线方向，计算光线从给定出射方向射出的概率，法线方向已被处理至与光线入射方向夹角大于90度
Float RoughDielectric::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
	auto eta_inv = inside ? eta_ : eta_inv_; //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比

	auto [alpha_u, alpha_v] = GetAlpha(texcoord);

	auto cos_i_n = glm::dot(-wi, normal),
		 cos_o_n = glm::dot(wo, normal);

	auto h = Vector3(0);
	auto relfect = cos_o_n > 0;
	if (relfect)
		h = glm::normalize(-wi + wo);
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
		auto jacobian = std::abs(1.0 / (4.0 * glm::dot(wo, h)));
		return F * D * jacobian;
	}
	else
	{
		auto jacobian = std::abs(glm::dot(wo, h) /
								 Sqr(eta_inv * glm::dot(-wi, h) + glm::dot(wo, h)));
		return (1.0 - F) * D * jacobian;
	}
}

///\brief 是否映射纹理
bool RoughDielectric::TextureMapping() const
{
	return Material::TextureMapping() ||
		   specular_reflectance_ && !specular_reflectance_->Constant() ||
		   specular_transmittance_ && !specular_transmittance_->Constant();
}

///\brief 补偿多次散射后又射出的光能
Float RoughDielectric::EvalMultipleScatter(Float cos_i_n, Float cos_o_n, bool inside) const
{
	auto f_add = inside ? f_add_inv_ : f_add_;
	auto albedo_i = GetAlbedo(std::abs(cos_i_n));
	auto albedo_o = GetAlbedo(std::abs(cos_o_n));
	auto f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
	return f_ms * f_add;
}
NAMESPACE_END(simple_renderer)