#ifndef CSRT__RENDERER__TEXTURES__TEXTURE_HPP
#define CSRT__RENDERER__TEXTURES__TEXTURE_HPP

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "bitmap.hpp"
#include "checkerboard.hpp"
#include "constant_texture.hpp"

namespace csrt
{

enum class TextureType
{
    kNone,
    kConstant,
    kCheckerboard,
    kBitmap,
};

struct TextureInfo
{
    TextureType type = TextureType::kNone;
    ConstantTextureData constant = {};
    CheckerboardData checkerboard = {};
    BitmapInfo bitmap = {};
};

struct TextureData
{
    TextureType type;
    union
    {
        ConstantTextureData constant;
        CheckerboardData checkerboard;
        BitmapData bitmap;
    };

    QUALIFIER_D_H TextureData();
    QUALIFIER_D_H TextureData(const TextureData &data);
    QUALIFIER_D_H ~TextureData() {}
    QUALIFIER_D_H void operator=(const TextureData &info);
};

class Texture
{
public:
    QUALIFIER_D_H Texture();
    QUALIFIER_D_H Texture(const uint32_t id, const TextureData &data,
                          const uint64_t pixel_offset);

    QUALIFIER_D_H Vec3 GetColor(const Vec2 &texcoord) const;
    QUALIFIER_D_H Vec2 GetGradient(const Vec2 &texcoord) const;
    QUALIFIER_D_H bool IsTransparent(const Vec2 &texcoord,
                                     uint32_t *seed) const;

private:
    uint64_t id_;
    TextureData data_;
};

} // namespace csrt

#endif