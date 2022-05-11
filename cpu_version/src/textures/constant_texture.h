#pragma once

#include <string>

#include "../core/texture_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 恒定颜色纹理派生类
class ConstantTexture : public Texture
{
public:
    ///\brief 恒定颜色纹理
    ConstantTexture(const Spectrum &color);

    ///\brief 恒定颜色纹理
    ConstantTexture(Float luminance);

    ///\return 纹理在给定坐标处像素值
    Spectrum Color(const Vector2 &coord) const override;

    ///\return 纹理在给定坐标处梯度
    Vector2 Gradient(const Vector2 &coord) const override;

private:
    Spectrum color_; //颜色
};

NAMESPACE_END(raytracer)