#include "csrt/renderer/texture.cuh"

#include <exception>

namespace csrt
{

QUALIFIER_D_H Texture::Data::Data()
    : type(Texture::Type::kNone), constant{}, checkerboard{}, bitmap{}
{
}

QUALIFIER_D_H Texture::Data::Data(const Texture::Data &info) : type(info.type)
{
    switch (info.type)
    {
    case Texture::Type::kNone:
        break;
    case Texture::Type::kConstant:
        constant = info.constant;
        break;
    case Texture::Type::kCheckerboard:
        checkerboard = info.checkerboard;
        break;
    case Texture::Type::kBitmap1:
    case Texture::Type::kBitmap3:
    case Texture::Type::kBitmap4:
        bitmap = info.bitmap;
        break;
    }
}

QUALIFIER_D_H void Texture::Data::operator=(const Texture::Data &info)
{
    type = info.type;
    switch (info.type)
    {
    case Texture::Type::kNone:
        break;
    case Texture::Type::kConstant:
        constant = info.constant;
        break;
    case Texture::Type::kCheckerboard:
        checkerboard = info.checkerboard;
        break;
    case Texture::Type::kBitmap1:
    case Texture::Type::kBitmap3:
    case Texture::Type::kBitmap4:
        bitmap = info.bitmap;
        break;
    }
}

Texture::Info::Info()
    : type(Texture::Type::kNone), constant{}, checkerboard{}, bitmap{}
{
}

Texture::Info::Info(const Texture::Info &info) : type(info.type)
{
    switch (info.type)
    {
    case Texture::Type::kNone:
        break;
    case Texture::Type::kConstant:
        constant = info.constant;
        break;
    case Texture::Type::kCheckerboard:
        checkerboard = info.checkerboard;
        break;
    case Texture::Type::kBitmap1:
    case Texture::Type::kBitmap3:
    case Texture::Type::kBitmap4:
        bitmap = info.bitmap;
        break;
    default:
        throw std::exception("unknow texture type.");
        break;
    }
}

void Texture::Info::operator=(const Texture::Info &info)
{
    type = info.type;
    switch (info.type)
    {
    case Texture::Type::kNone:
        break;
    case Texture::Type::kConstant:
        constant = info.constant;
        break;
    case Texture::Type::kCheckerboard:
        checkerboard = info.checkerboard;
        break;
    case Texture::Type::kBitmap1:
    case Texture::Type::kBitmap3:
    case Texture::Type::kBitmap4:
        bitmap = info.bitmap;
        break;
    default:
        throw std::exception("unknow texture type.");
        break;
    }
}

QUALIFIER_D_H Texture::Texture() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Texture::Texture(const uint32_t id, const Texture::Data &data,
                               const uint64_t offset_data)
    : id_(id), data_(data)
{
    if (data_.type == Texture::Type::kBitmap1 ||
        data_.type == Texture::Type::kBitmap3 ||
        data_.type == Texture::Type::kBitmap4)
    {
        data_.bitmap.data = data_.bitmap.data + offset_data;
    }
}

QUALIFIER_D_H Vec3 Texture::GetColor(const Vec2 &texcoord) const
{
    switch (data_.type)
    {
    case Texture::Type::kConstant:
        return data_.constant.color;
        break;
    case Texture::Type::kCheckerboard:
        return GetColorCheckerboard(texcoord);
        break;
    case Texture::Type::kBitmap1:
        return {GetColorBitmap1(texcoord)};
        break;
    case Texture::Type::kBitmap3:
        return GetColorBitmap<3>(texcoord);
        break;
    case Texture::Type::kBitmap4:
        return GetColorBitmap<4>(texcoord);
        break;
    }
    return {};
}

QUALIFIER_D_H Vec2 Texture::GetGradient(const Vec2 &texcoord) const
{
    switch (data_.type)
    {
    case Texture::Type::kConstant:
        return {};
        break;
    case Texture::Type::kCheckerboard:
        return GetGradientCheckerboard(texcoord);
        break;
    case Texture::Type::kBitmap1:
        return GetGradientBitmap1(texcoord);
        break;
    case Texture::Type::kBitmap3:
        return GetGradientBitmap<3>(texcoord);
        break;
    case Texture::Type::kBitmap4:
        return GetGradientBitmap<4>(texcoord);
        break;
    }
    return {};
}

QUALIFIER_D_H bool Texture::IsTransparent(const Vec2 &texcoord,
                                          uint32_t *seed) const
{
    switch (data_.type)
    {
    case Texture::Type::kConstant:
        return data_.constant.color.x < RandomFloat(seed);
        break;
    case Texture::Type::kBitmap1:
        return GetColorBitmap1(texcoord) < RandomFloat(seed);
        break;
    case Texture::Type::kBitmap4:
        return IsTransparentBitmap4(texcoord, seed);
        break;
    }
    return false;
}

} // namespace csrt
