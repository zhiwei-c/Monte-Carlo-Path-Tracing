#pragma once

#include <string>

#include "../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

inline Float GetLuminance(const Vector3 &rgb)
{
    return std::max(std::max(rgb.r, rgb.g), rgb.b);
}

//纹理类型
enum class TextureType
{
    kConstantTexture, //常量颜色
    kBitmap,          //位图
    kCheckerboard,    //棋盘图
    kGridTexture,     //网格图
};

//纹理类
class Texture
{
public:
    virtual ~Texture() {}

    /**
     * \brief 纹理在给定坐标处像素值
     * \param texcoord 纹理坐标
     * \return 像素值
     */
    virtual Spectrum GetPixel(const Vector2 &coord) const = 0;

    /**
     * \brief 纹理在给定坐标处梯度
     * \param texcoord 纹理坐标
     * \return 梯度
     */
    virtual Vector2 GetGradient(const Vector2 &coord) const = 0;

    /**
     * \brief 材质在给定的纹理坐标处是否透明
     * \param texcoord 纹理坐标
     */
    virtual bool Transparent(const Vector2 &coord) const { return false; }

    bool Constant() const { return type_ == TextureType::kConstantTexture; }

protected:
    Texture(TextureType type) : type_(type) {}

private:
    TextureType type_; //纹理类型
};

NAMESPACE_END(simple_renderer)