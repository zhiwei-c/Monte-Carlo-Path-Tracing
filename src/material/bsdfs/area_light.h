#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

//面光源
class AreaLight : public Material
{
public:
	AreaLight(const std::string &name, const Vector3 &ke) : Material(name, MaterialType::kAreaLight), radiance_(ke){};

	Vector3 radiance() const override { return radiance_; }

	bool HasEmission() const override { return true; }

	std::pair<Vector3, BsdfSamplingType> Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override { return {Vector3(0), BsdfSamplingType::kNone}; }

	Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override { return Vector3(0); }

	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, const BsdfSamplingType &bsdf_sampling_type) const override { return 0; };

private:
	Vector3 radiance_; //辐射亮度
};

NAMESPACE_END(simple_renderer)