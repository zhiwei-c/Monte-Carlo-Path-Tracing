#pragma once

#include "texture.hpp"

NAMESPACE_BEGIN(raytracer)

//纹理派生类，棋盘
class Checkerboard : public Texture
{
public:
    Checkerboard(const std::string &id, Texture *color0, Texture *color1, dmat3 to_uv);

    dvec3 color(const dvec2 &texcoord) const override;
    dvec2 gradient(const dvec2 &texcoord) const override;

private:
    Texture *color0_; //棋盘上交错的颜色
    Texture *color1_; //棋盘上交错的颜色
    dmat3 to_uv_;     //纹理坐标变换矩阵
};

NAMESPACE_END(raytracer)