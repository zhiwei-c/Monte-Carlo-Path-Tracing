#include "texture.cuh"

namespace rt
{

QUALIFIER_D_H float Texture::GetColorBitmap1(const Vec2 &texcoord) const
{
    const Vec3 uv = TransformPoint(data_.checkerboard.to_uv, {texcoord, 0.0f});
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

    const float color_00 = data_.bitmap.data[data_.bitmap.offset +
                                             (x_0 + data_.bitmap.width * y_0)],
                color_01 = data_.bitmap.data[data_.bitmap.offset +
                                             (x_0 + data_.bitmap.width * y_1)],
                color_10 = data_.bitmap.data[data_.bitmap.offset +
                                             (x_1 + data_.bitmap.width * y_0)],
                color_11 = data_.bitmap.data[data_.bitmap.offset +
                                             (x_1 + data_.bitmap.width * y_1)];
    const float color_0 = Lerp(color_00, color_01, t_y),
                color_1 = Lerp(color_10, color_11, t_y);
    return Lerp(color_0, color_1, t_x);
}

QUALIFIER_D_H Vec2 Texture::GetGradientBitmap1(const Vec2 &texcoord) const
{
    constexpr float delta = 1e-4f, norm = 1.0f / delta;
    const float value = GetColorBitmap1(texcoord),
                value_u = GetColorBitmap1(texcoord + Vec2{delta, 0}),
                value_v = GetColorBitmap1(texcoord + Vec2{0, delta});
    return {(value_u - value) * norm, (value_v - value) * norm};
}

QUALIFIER_D_H bool Texture::IsTransparentBitmap4(const Vec2 &texcoord,
                                                 const float xi) const
{
    const Vec3 uv = TransformPoint(data_.checkerboard.to_uv, {texcoord, 0.0f});
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

    uint64_t offset =
        data_.bitmap.offset + (x_0 + data_.bitmap.width * y_0) * 4 + 3;
    const float color_00 = data_.bitmap.data[offset];

    offset = data_.bitmap.offset + (x_0 + data_.bitmap.width * y_1) * 4 + 3;
    const float color_01 = data_.bitmap.data[offset];

    offset = data_.bitmap.offset + (x_1 + data_.bitmap.width * y_0) * 4 + 3;
    const float color_10 = data_.bitmap.data[offset];

    offset = data_.bitmap.offset + (x_1 + data_.bitmap.width * y_1) * 4 + 3;
    const float color_11 = data_.bitmap.data[offset];

    const float color_0 = Lerp(color_00, color_01, t_y),
                color_1 = Lerp(color_10, color_11, t_y);
    return Lerp(color_0, color_1, t_x) > xi;
}

} // namespace rt