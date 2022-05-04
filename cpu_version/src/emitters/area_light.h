#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 面光源派生类
class AreaLight : public Material
{
public:
	///\brief 面光源
	///\param radiance 辐射亮度
	AreaLight(const Spectrum &radiance)
		: Material(MaterialType::kAreaLight),
		  radiance_(radiance){};

	///\return 辐射亮度
	Spectrum radiance() const override { return radiance_; }

private:
	Spectrum radiance_; //辐射亮度
};

NAMESPACE_END(simple_renderer)