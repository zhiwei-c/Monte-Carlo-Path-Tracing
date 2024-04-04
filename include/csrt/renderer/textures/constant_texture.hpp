#ifndef CSRT__RENDERER__TEXTURES__CONSTANT_TEXTURE_HPP
#define CSRT__RENDERER__TEXTURES__CONSTANT_TEXTURE_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"

namespace csrt
{

struct ConstantTextureData
{
    Vec3 color = {0.5f};
};

QUALIFIER_D_H Vec3 GetColorConstantTexture(const ConstantTextureData &data,
                                           const Vec2 &texcoord);

QUALIFIER_D_H Vec2 GetGradientConstantTexture(const ConstantTextureData &data,
                                              const Vec2 &texcoord);

QUALIFIER_D_H bool IsTransparentConstantTexture(const ConstantTextureData &data,
                                                const Vec2 &texcoord,
                                                uint32_t *seed);

} // namespace csrt

#endif