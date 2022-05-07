#include <algorithm>

#include "rough_conductor.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 粗糙的导体材质
RoughConductor::RoughConductor(bool mirror,
							   const Spectrum &eta,
							   const Spectrum &k,
							   Float ext_ior,
							   std::unique_ptr<Texture> specular_reflectance,
							   MicrofacetDistribType distrib_type,
							   std::unique_ptr<Texture> alpha_u,
							   std::unique_ptr<Texture> alpha_v)
	: Material(MaterialType::kRoughConductor),
	  Microfacet(distrib_type,
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

	if (Material::TextureMapping())
	{
		albedo_avg_ = -1;
		return;
	}

	ComputeAlbedoTable();
	if (albedo_avg_ < 0)
		return;
	auto [reflectivity, edgetint] = IorToReflectivityEdgetint(eta_, k_);
	auto F_avg = AverageFresnelConductor(reflectivity, edgetint);
	f_add_ = Sqr(F_avg) * albedo_avg_ / (Spectrum(1) - F_avg * (1 - albedo_avg_));
}

///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
void RoughConductor::Sample(BsdfSampling &bs) const
{
	auto [alpha_u, alpha_v] = GetAlpha(bs.texcoord);

	auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
	auto [h, D] = distrib->Sample(bs.normal, {UniformFloat(), UniformFloat()});

	if (D < kEpsilon)
		return;

	bs.wi = -Reflect(-bs.wo, h);
	auto cos_i_n = glm::dot(bs.wi, bs.normal);
	if (cos_i_n >= 0)
		return;

	auto jacobian = std::abs(1.0 / (4.0 * glm::dot(bs.wo, h)));
	bs.pdf = jacobian * D;
	if (bs.pdf < kEpsilonL)
	{
		bs.pdf = 0;
		return;
	}
	if (!bs.get_attenuation)
		return;

	auto cos_o_n = glm::dot(bs.wo, bs.normal);
	auto F = mirror_ ? Spectrum(1) : FresnelConductor(bs.wi, h, eta_, k_);
	auto G = distrib->SmithG1(-bs.wi, h, bs.normal) * distrib->SmithG1(bs.wo, h, bs.normal);
	auto albedo = F * static_cast<Float>(D * G / std::abs(4.0 * cos_i_n * cos_o_n));
	if (specular_reflectance_)
		albedo *= specular_reflectance_->Color(bs.texcoord);

	if (albedo_avg_ > 0)
		albedo += EvalMultipleScatter(cos_i_n, cos_o_n);

	bs.attenuation = albedo;
}

///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
Spectrum RoughConductor::Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
	auto [alpha_u, alpha_v] = GetAlpha(texcoord);
	auto cos_i_n = glm::dot(wi, normal),
		 cos_o_n = glm::dot(wo, normal);

	auto h = glm::normalize(-wi + wo);
	auto F = mirror_ ? Spectrum(1) : FresnelConductor(wi, h, eta_, k_);

	auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
	auto D = distrib->Pdf(h, normal);
	auto G = distrib->SmithG1(-wi, h, normal) * distrib->SmithG1(wo, h, normal);

	auto albedo = F * static_cast<Float>(D * G / std::abs(4.0 * cos_i_n * cos_o_n));

	if (specular_reflectance_)
		albedo *= specular_reflectance_->Color(texcoord);

	if (albedo_avg_ > 0)
		albedo += EvalMultipleScatter(cos_i_n, cos_o_n);

	return albedo;
}

///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
Float RoughConductor::Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const
{
    // 入射、出射光线需在同侧
	if (NotSameHemis(wo, -wi))
		return 0;

	auto [alpha_u, alpha_v] = GetAlpha(texcoord);
	auto h = glm::normalize(-wi + wo);

	auto distrib = InitDistrib(distrib_type_, alpha_u, alpha_v);
	auto D = distrib->Pdf(h, normal);
	if (D < kEpsilon)
		return 0;

	auto jacobian = std::abs(1.0 / (4.0 * glm::dot(wo, h)));
	return D * jacobian;
}

///\brief 是否映射纹理
bool RoughConductor::TextureMapping() const
{
	return Material::TextureMapping() ||
		   specular_reflectance_ && !specular_reflectance_->Constant();
}

///\brief 补偿多次散射后又射出的光能
Spectrum RoughConductor::EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
{
	auto albedo_i = GetAlbedo(std::abs(cos_i_n));
	auto albedo_o = GetAlbedo(std::abs(cos_o_n));
	auto f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
	return f_ms * f_add_;
}

NAMESPACE_END(simple_renderer)