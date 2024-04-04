#include "csrt/renderer/textures/bitmap.hpp"

namespace csrt
{

QUALIFIER_D_H Vec3 GetColorBitmap(const BitmapData &data, const Vec2 &texcoord)
{
    const Vec3 uv = TransformPoint(data.to_uv, {texcoord, 0.0f});
    float x = uv.x * data.width, y = uv.y * data.height;
    while (x < 0)
        x += data.width;
    while (x > data.width - 1)
        x -= data.width;
    while (y < 0)
        y += data.height;
    while (y > data.height - 1)
        y -= data.height;

    const uint32_t x_0 = static_cast<uint32_t>(x),
                   y_0 = static_cast<uint32_t>(y);
    const float t_x = x - x_0, t_y = y - y_0;
    const uint32_t x_1 = (t_x > 0.0f) ? x_0 + 1 : x_0,
                   y_1 = (t_y > 0.0f) ? y_0 + 1 : y_0;
    if (data.channel == 1)
    {
        const float color_00 = data.data[(x_0 + data.width * y_0)],
                    color_01 = data.data[(x_0 + data.width * y_1)],
                    color_10 = data.data[(x_1 + data.width * y_0)],
                    color_11 = data.data[(x_1 + data.width * y_1)];
        const float color_0 = Lerp(color_00, color_01, t_y),
                    color_1 = Lerp(color_10, color_11, t_y);
        return Lerp(color_0, color_1, t_x);
    }
    else
    {
        uint32_t offset = (x_0 + data.width * y_0) * data.channel;
        const Vec3 color_00 = {data.data[offset], data.data[offset + 1],
                               data.data[offset + 2]};

        offset = (x_0 + data.width * y_1) * data.channel;
        const Vec3 color_01 = {data.data[offset], data.data[offset + 1],
                               data.data[offset + 2]};

        offset = (x_1 + data.width * y_0) * data.channel;
        const Vec3 color_10 = {data.data[offset], data.data[offset + 1],
                               data.data[offset + 2]};

        offset = (x_1 + data.width * y_1) * data.channel;
        const Vec3 color_11 = {data.data[offset], data.data[offset + 1],
                               data.data[offset + 2]};

        const Vec3 color_0 = Lerp(color_00, color_01, t_y),
                   color_1 = Lerp(color_10, color_11, t_y);
        return Lerp(color_0, color_1, t_x);
    }
}

QUALIFIER_D_H Vec2 GetGradientBitmap(const BitmapData &data,
                                     const Vec2 &texcoord)
{
    constexpr float delta = 1e-4f, norm = 1.0f / delta;
    const float value = Length(GetColorBitmap(data, texcoord)),
                value_u =
                    Length(GetColorBitmap(data, texcoord + Vec2{delta, 0})),
                value_v =
                    Length(GetColorBitmap(data, texcoord + Vec2{0, delta}));
    return {(value_u - value) * norm, (value_v - value) * norm};
}

QUALIFIER_D_H bool IsTransparentBitmap(const BitmapData &data,
                                       const Vec2 &texcoord, uint32_t *seed)
{
    if (data.channel != 4)
        return false;

    const Vec3 uv = TransformPoint(data.to_uv, {texcoord, 0.0f});
    float x = uv.x * data.width, y = uv.y * data.height;
    while (x < 0)
        x += data.width;
    while (x > data.width - 1)
        x -= data.width;
    while (y < 0)
        y += data.height;
    while (y > data.height - 1)
        y -= data.height;

    const uint32_t x_0 = static_cast<uint32_t>(x),
                   y_0 = static_cast<uint32_t>(y);
    const float t_x = x - x_0, t_y = y - y_0;
    const uint32_t x_1 = (t_x > 0.0f) ? x_0 + 1 : x_0,
                   y_1 = (t_y > 0.0f) ? y_0 + 1 : y_0;

    const float color_00 = data.data[(x_0 + data.width * y_0) * 4 + 3],
                color_01 = data.data[(x_0 + data.width * y_1) * 4 + 3],
                color_10 = data.data[(x_1 + data.width * y_0) * 4 + 3],
                color_11 = data.data[(x_1 + data.width * y_1) * 4 + 3];

    const float color_0 = Lerp(color_00, color_01, t_y),
                color_1 = Lerp(color_10, color_11, t_y);
    return Lerp(color_0, color_1, t_x) < RandomFloat(seed);
}

} // namespace csrt