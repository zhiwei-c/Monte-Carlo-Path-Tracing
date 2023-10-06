#include "constant_texture.hpp"

NAMESPACE_BEGIN(raytracer)

ConstantTexture::ConstantTexture(const std::string &id, const dvec3 &color)
    : Texture(TextureType::kConstant, id),
      color_(color)
{
}

ConstantTexture::ConstantTexture(const std::string &id, double value)
    : Texture(TextureType::kConstant, id),
      color_(dvec3(value))
{
}

dvec3 ConstantTexture::color(const dvec2 &texcoord) const
{
    return color_;
}

dvec2 ConstantTexture::gradient(const dvec2 &texcoord) const
{
    return {0, 0};
}

NAMESPACE_END(raytracer)