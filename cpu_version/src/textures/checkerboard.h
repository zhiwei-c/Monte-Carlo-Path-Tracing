#pragma once

#include <string>

#include "../core/texture_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 过程式棋盘纹理派生类
class Checkerboard : public Texture
{
public:
    //过程式棋盘纹理
    Checkerboard(const Spectrum &color0, const Spectrum &color1, const Vector2 &uv_offset, const Vector2 &uv_scale);

    ///\return 纹理在给定坐标处像素值
    Spectrum Color(const Vector2 &coord) const override;

    ///\return 纹理在给定坐标处梯度
    Vector2 Gradient(const Vector2 &coord) const override;

private:
    Spectrum color0_, color1_;           //面片颜色
    std::unique_ptr<Vector2> uv_offset_; //纹理坐标偏置
    std::unique_ptr<Vector2> uv_scale_;  //纹理坐标偏比例
};

NAMESPACE_END(simple_renderer)