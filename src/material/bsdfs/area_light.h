#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

//面光源
class AreaLight : public Material
{
public:
	AreaLight(const std::string &name, const Spectrum &radiance) : Material(name, MaterialType::kAreaLight), radiance_(radiance){};

	Spectrum radiance() const override { return radiance_; }

	bool HasEmission() const override { return true; }

	BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return BsdfSampling(); }

	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return Spectrum(0); }

	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return 0; };

private:
	Spectrum radiance_; //辐射亮度
};

NAMESPACE_END(simple_renderer)