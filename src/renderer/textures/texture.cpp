#include "csrt/renderer/textures/texture.hpp"

#include <exception>

namespace csrt
{

QUALIFIER_D_H TextureData::TextureData()
    : type(TextureType::kNone), constant{}
{
}

QUALIFIER_D_H TextureData::TextureData(const TextureData &data)
    : type(data.type)
{
    switch (data.type)
    {
    case TextureType::kNone:
        break;
    case TextureType::kConstant:
        constant = data.constant;
        break;
    case TextureType::kCheckerboard:
        checkerboard = data.checkerboard;
        break;
    case TextureType::kBitmap:
        bitmap = data.bitmap;
        break;
    }
}

QUALIFIER_D_H void TextureData::operator=(const TextureData &data)
{
    type = data.type;
    switch (data.type)
    {
    case TextureType::kNone:
        break;
    case TextureType::kConstant:
        constant = data.constant;
        break;
    case TextureType::kCheckerboard:
        checkerboard = data.checkerboard;
        break;
    case TextureType::kBitmap:
        bitmap = data.bitmap;
        break;
    }
}

QUALIFIER_D_H Texture::Texture() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Texture::Texture(const uint32_t id, const TextureData &data,
                               const uint64_t pixel_offset)
    : id_(id), data_(data)
{
    if (data_.type == TextureType::kBitmap)
    {
        data_.bitmap.data = data_.bitmap.data + pixel_offset;
    }
}

QUALIFIER_D_H Vec3 Texture::GetColor(const Vec2 &texcoord) const
{
    switch (data_.type)
    {
    case TextureType::kConstant:
        return GetColorConstantTexture(data_.constant, texcoord);
        break;
    case TextureType::kCheckerboard:
        return GetColorCheckerboard(data_.checkerboard, texcoord);
        break;
    case TextureType::kBitmap:
        return GetColorBitmap(data_.bitmap, texcoord);
        break;
    }
    return {};
}

QUALIFIER_D_H Vec2 Texture::GetGradient(const Vec2 &texcoord) const
{
    switch (data_.type)
    {
    case TextureType::kConstant:
        return GetGradientConstantTexture(data_.constant, texcoord);
        break;
    case TextureType::kCheckerboard:
        return GetGradientCheckerboard(data_.checkerboard, texcoord);
        break;
    case TextureType::kBitmap:
        return GetGradientBitmap(data_.bitmap, texcoord);
        break;
    }
    return {};
}

QUALIFIER_D_H bool Texture::IsTransparent(const Vec2 &texcoord,
                                          uint32_t *seed) const
{
    switch (data_.type)
    {
    case TextureType::kConstant:
        return IsTransparentConstantTexture(data_.constant, texcoord, seed);
        break;
    case TextureType::kCheckerboard:
        return IsTransparentCheckerboard(data_.checkerboard, texcoord, seed);
        break;
    case TextureType::kBitmap:
        return IsTransparentBitmap(data_.bitmap, texcoord, seed);
        break;
    }
    return false;
}

} // namespace csrt
