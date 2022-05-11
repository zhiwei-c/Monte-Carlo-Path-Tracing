#pragma once

#include <string>

#include "../utils/math.h"

NAMESPACE_BEGIN(raytracer)

//纹理类型
enum class TextureType
{
    kConstantTexture, //恒定颜色纹理
    kBitmap,          //位图
    kCheckerboard,    //过程式棋盘纹理
    kGridTexture,     //过程式网格纹理
};

//纹理基类
class Texture
{
public:
    virtual ~Texture() {}

    ///\return 纹理在给定坐标处像素值
    virtual Spectrum Color(const Vector2 &coord) const = 0;

    ///\return 纹理在给定坐标处梯度
    virtual Vector2 Gradient(const Vector2 &coord) const = 0;

    ///\return 材质在给定的纹理坐标处是否透明
    virtual bool Transparent(const Vector2 &coord) const { return false; }

    ///\return 纹理的颜色是否随纹理坐标的改变而改变
    bool Constant() const { return type_ == TextureType::kConstantTexture; }

protected:
    Texture(TextureType type) : type_(type) {}

private:
    TextureType type_; //纹理类型
};

NAMESPACE_END(raytracer)