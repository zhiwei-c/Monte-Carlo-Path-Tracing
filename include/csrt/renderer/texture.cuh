#pragma once

#include "../tensor.cuh"
#include "../utils.cuh"

namespace csrt
{

class Texture
{
public:
    enum Type
    {
        kNone,
        kConstant,
        kCheckerboard,
        kBitmap1,
        kBitmap3,
        kBitmap4,
    };

    struct Data
    {
        struct Constant
        {
            Vec3 color = {0.5f};
        };

        struct Checkerboard
        {
            Vec3 color0 = {0.4f};
            Vec3 color1 = {0.2f};
            Mat4 to_uv = {};
        };

        struct Bitmap
        {
            int width = 0;
            int height = 0;
            float *data = nullptr;
            Mat4 to_uv = {};
        };

        Texture::Type type;
        union
        {
            Constant constant;
            Checkerboard checkerboard;
            Bitmap bitmap;
        };

        QUALIFIER_D_H Data();
        QUALIFIER_D_H ~Data() {}
        QUALIFIER_D_H Data(const Texture::Data &info);
        QUALIFIER_D_H void operator=(const Texture::Data &info);
    };

    struct Info
    {
        Texture::Type type;
        union
        {
            Texture::Data::Constant constant;
            Texture::Data::Checkerboard checkerboard;
        };
        struct Bitmap
        {
            int width = 0;
            int height = 0;
            std::vector<float> data;
            Mat4 to_uv = {};
        } bitmap;

        Info();
        ~Info() {}
        Info(const Info &info);
        void operator=(const Info &info);
    };

    QUALIFIER_D_H Texture();
    QUALIFIER_D_H Texture(const uint32_t id, const Texture::Data &data,
                          const uint64_t offset_data);

    QUALIFIER_D_H Vec3 GetColor(const Vec2 &texcoord) const;
    QUALIFIER_D_H Vec2 GetGradient(const Vec2 &texcoord) const;
    QUALIFIER_D_H bool IsTransparent(const Vec2 &texcoord,
                                     uint32_t *seed) const;

private:
    QUALIFIER_D_H Vec3 GetColorCheckerboard(const Vec2 &texcoord) const;
    QUALIFIER_D_H float GetColorBitmap1(const Vec2 &texcoord) const;
    template <int channel>
    QUALIFIER_D_H Vec3 GetColorBitmap(const Vec2 &texcoord) const;

    QUALIFIER_D_H Vec2 GetGradientCheckerboard(const Vec2 &texcoord) const;
    QUALIFIER_D_H Vec2 GetGradientBitmap1(const Vec2 &texcoord) const;
    template <int channel>
    QUALIFIER_D_H Vec2 GetGradientBitmap(const Vec2 &texcoord) const;

    QUALIFIER_D_H bool IsTransparentBitmap4(const Vec2 &texcoord,
                                            uint32_t *seed) const;

    uint64_t id_;
    Data data_;
};

template <int channel>
QUALIFIER_D_H inline Vec3 Texture::GetColorBitmap(const Vec2 &texcoord) const
{
    const Vec3 uv = TransformPoint(data_.bitmap.to_uv, {texcoord, 0.0f});
    float x = uv.x * data_.bitmap.width, y = uv.y * data_.bitmap.height;
    while (x < 0)
        x += data_.bitmap.width;
    while (x > data_.bitmap.width - 1)
        x -= data_.bitmap.width;
    while (y < 0)
        y += data_.bitmap.height;
    while (y > data_.bitmap.height - 1)
        y -= data_.bitmap.height;

    const uint32_t x_0 = static_cast<uint32_t>(x),
                   y_0 = static_cast<uint32_t>(y);
    const float t_x = x - x_0, t_y = y - y_0;
    const uint32_t x_1 = (t_x > 0.0f) ? x_0 + 1 : x_0,
                   y_1 = (t_y > 0.0f) ? y_0 + 1 : y_0;

    uint32_t offset = (x_0 + data_.bitmap.width * y_0) * channel;
    const Vec3 color_00 = {data_.bitmap.data[offset],
                           data_.bitmap.data[offset + 1],
                           data_.bitmap.data[offset + 2]};

    offset = (x_0 + data_.bitmap.width * y_1) * channel;
    const Vec3 color_01 = {data_.bitmap.data[offset],
                           data_.bitmap.data[offset + 1],
                           data_.bitmap.data[offset + 2]};

    offset = (x_1 + data_.bitmap.width * y_0) * channel;
    const Vec3 color_10 = {data_.bitmap.data[offset],
                           data_.bitmap.data[offset + 1],
                           data_.bitmap.data[offset + 2]};

    offset = (x_1 + data_.bitmap.width * y_1) * channel;
    const Vec3 color_11 = {data_.bitmap.data[offset],
                           data_.bitmap.data[offset + 1],
                           data_.bitmap.data[offset + 2]};

    const Vec3 color_0 = Lerp(color_00, color_01, t_y),
               color_1 = Lerp(color_10, color_11, t_y);
    return Lerp(color_0, color_1, t_x);
}

template <int channel>
QUALIFIER_D_H inline Vec2 Texture::GetGradientBitmap(const Vec2 &texcoord) const
{
    constexpr float delta = 1e-4f, norm = 1.0f / delta;
    const float value = Length(GetColorBitmap<channel>(texcoord)),
                value_u =
                    Length(GetColorBitmap<channel>(texcoord + Vec2{delta, 0})),
                value_v =
                    Length(GetColorBitmap<channel>(texcoord + Vec2{0, delta}));
    return {(value_u - value) * norm, (value_v - value) * norm};
}

} // namespace csrt
