#include "constant_texture.h"

NAMESPACE_BEGIN(raytracer)

///\brief 恒定颜色纹理
ConstantTexture::ConstantTexture(const Spectrum &color)
    : Texture(TextureType::kConstantTexture), color_(color)
{
}

///\brief 恒定颜色纹理
ConstantTexture::ConstantTexture(Float luminance)
    : Texture(TextureType::kConstantTexture), color_(Spectrum(luminance))
{
}

///\return 纹理在给定坐标处像素值
Spectrum ConstantTexture::Color(const Vector2 &coord) const
{
    return color_;
}

///\return 纹理在给定坐标处梯度
Vector2 ConstantTexture::Gradient(const Vector2 &coord) const
{
    return Vector2(0, 0);
}

NAMESPACE_END(raytracer)