#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

//面光源
class AreaLight : public Material
{
public:
	AreaLight(const std::string &name, const Spectrum &radiance, const Float sampling_weight = 1)
		: Material(name, MaterialType::kAreaLight), radiance_(radiance), sampling_weight_(sampling_weight){};

    ///\return 辐射亮度
	Spectrum radiance() const override { return radiance_; }

	Float sampling_weight() const { return sampling_weight_; }

	bool HasEmission() const override { return true; }

	BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override { return BsdfSampling(); }

	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return Spectrum(0); }

	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return 0; };

private:
	Float sampling_weight_; //额外权重
	Spectrum radiance_;		//辐射亮度
};

NAMESPACE_END(simple_renderer)