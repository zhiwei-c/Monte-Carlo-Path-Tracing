#include "csrt/renderer/textures/constant_texture.cuh"

namespace csrt
{

QUALIFIER_D_H Vec3 GetColorConstantTexture(const ConstantTextureData &data,
                                           const Vec2 &texcoord)
{
    return data.color;
}

QUALIFIER_D_H Vec2 GetGradientConstantTexture(const ConstantTextureData &data,
                                              const Vec2 &texcoord)
{
    return {};
}

QUALIFIER_D_H bool IsTransparentConstantTexture(const ConstantTextureData &data,
                                                const Vec2 &texcoord,
                                                uint32_t *seed)
{
    return data.color.x < RandomFloat(seed);
}

} // namespace csrt