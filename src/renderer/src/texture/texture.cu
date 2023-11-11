#include "texture.cuh"

#include <exception>

namespace rt
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

Texture::Info Texture::Info::CreateConstant(const Vec3 &color)
{
    Texture::Info info;
    info.type = Texture::Type::kConstant;
    info.constant.color = color;
    return info;
}

Texture::Info Texture::Info::CreateCheckerboard(const Vec3 &color0,
                                                const Vec3 &color1,
                                                const Mat4 &to_uv)
{
    Texture::Info info;
    info.type = Texture::Type::kCheckerboard;
    info.checkerboard.color0 = color0;
    info.checkerboard.color1 = color1;
    info.checkerboard.to_uv = to_uv;
    return info;
}

Texture::Info Texture::Info::CreateBitmap(const int width, const int height,
                                          const int channel,
                                          const std::vector<float> &data,
                                          const Mat4 &to_uv)
{
    Texture::Info info;
    switch (channel)
    {
    case 1:
    {
        info.type = Texture::Type::kBitmap1;
        break;
    }
    case 3:
    {
        info.type = Texture::Type::kBitmap3;
        break;
    }
    case 4:
    {
        info.type = Texture::Type::kBitmap4;
        break;
    }
    default:
    {
        std::ostringstream oss;
        oss << "unsupport bitmap channel '" << channel << "'.";
        throw std::exception(oss.str().c_str());
        break;
    }
    }
    info.bitmap.width = width;
    info.bitmap.height = height;
    info.bitmap.data = data;
    return info;
}

QUALIFIER_D_H Texture::Texture() : id_(kInvalidId), data_{} {}

QUALIFIER_D_H Texture::Texture(const uint32_t id, const Texture::Data &data)
    : id_(id), data_(data)
{
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

QUALIFIER_D_H Vec2 rt::Texture::GetGradient(const Vec2 &texcoord) const
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
                                          const float xi) const
{
    switch (data_.type)
    {
    case Texture::Type::kConstant:
        return data_.constant.color.x > xi;
        break;
    case Texture::Type::kBitmap1:
        return GetColorBitmap1(texcoord) > xi;
        break;
    case Texture::Type::kBitmap4:
        return IsTransparentBitmap4(texcoord, xi);
        break;
    }
    return false;
}

} // namespace rt
