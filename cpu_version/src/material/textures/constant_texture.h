#pragma once

#include <string>

#include "../texture.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

//棋盘图
class ConstantTexture : public Texture
{
public:
    //棋盘图
    ConstantTexture(const Spectrum &color)
        : Texture(TextureType::kConstantTexture),
          color_(color) {}

    ConstantTexture(const Float &luminance)
        : Texture(TextureType::kConstantTexture),
          color_(Spectrum(luminance)) {}

    Spectrum GetPixel(const Vector2 &coord) const override
    {
        return color_;
    }

    Vector2 GetGradient(const Vector2 &coord) const override
    {
        return Vector2(0, 0);
    }

private:
    Spectrum color_;
};

NAMESPACE_END(simple_renderer)