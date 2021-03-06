#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 面光源派生类
class AreaLight : public Bsdf
{
public:
	///\brief 面光源
	///\param radiance 辐射亮度
	AreaLight(const Spectrum &radiance) : Bsdf(BsdfType::kAreaLight), radiance_(radiance){};

	///\return 辐射亮度
	Spectrum radiance() const override { return radiance_; }

private:
	Spectrum radiance_; //辐射亮度
};

NAMESPACE_END(raytracer)