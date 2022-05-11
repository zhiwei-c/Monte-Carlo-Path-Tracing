#pragma once

#include <string>

#include "../core/texture_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 过程式网格纹理派生类
class GridTexture : public Texture
{
public:
    ///\brief 过程式网格纹理
    GridTexture(const Spectrum &color0, const Spectrum &color1, Float line_width, const Vector2 &uv_offset, const Vector2 &uv_scale);

    ///\return 纹理在给定坐标处像素值
    Spectrum Color(const Vector2 &coord) const override;

    ///\return 纹理在给定坐标处梯度
    Vector2 Gradient(const Vector2 &coord) const override;

private:
    Float line_width_;                   // UV 空间下网格线的宽度
    Spectrum color0_;                    //纹理背景颜色
    Spectrum color1_;                    //网格线颜色
    std::unique_ptr<Vector2> uv_offset_; //纹理坐标偏置
    std::unique_ptr<Vector2> uv_scale_;  //纹理坐标偏比例
};

NAMESPACE_END(raytracer)