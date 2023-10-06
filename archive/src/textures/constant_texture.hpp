#pragma once

#include "texture.hpp"

NAMESPACE_BEGIN(raytracer)

//纹理派生类，恒定的颜色，
class ConstantTexture : public Texture
{
public:
    ConstantTexture(const std::string &id, const dvec3 &color);
    ConstantTexture(const std::string &id, double value);

    dvec3 color(const dvec2 &texcoord) const override;
    dvec2 gradient(const dvec2 &texcoord) const override;

private:
    dvec3 color_; //颜色
};

NAMESPACE_END(raytracer)