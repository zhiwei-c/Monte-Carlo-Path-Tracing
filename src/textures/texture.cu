#include "texture.cuh"

#include "../utils/math.cuh"

Texture::Info Texture::Info::CreateConstant(const Vec3 &color)
{
    Texture::Info info;
    info.type = Texture::Type::kConstant;
    info.data.constant.color = color;
    return info;
}

Texture::Info Texture::Info::CreateCheckerboard(const Vec3 &color0, const Vec3 &color1,
                                                const Mat4 &to_uv)
{
    Texture::Info info;
    info.type = Texture::Type::kCheckerboard;
    info.data.checkerboard.color0 = color0;
    info.data.checkerboard.color1 = color1;
    info.data.checkerboard.to_uv = to_uv;
    return info;
}

Texture::Info Texture::Info::CreateBitmap(const uint64_t offset, const int width,
                                          const int height, const int channel)
{
    Texture::Info info;
    info.type = Texture::Type::kBitmap;
    info.data.bitmap.offset = offset;
    info.data.bitmap.width = width;
    info.data.bitmap.height = height;
    info.data.bitmap.channel = channel;
    return info;
}

QUALIFIER_DEVICE Vec3 CheckerboardTexture::GetColor(const Vec2 &texcoord, const float *pixel_buffer) const
{
    Vec3 uv = TransfromPoint(to_uv_, {texcoord.u, texcoord.v, 0.0f});
    while (uv.x > 1)
        uv.x -= 1;
    while (uv.x < 0)
        uv.x += 1;
    while (uv.y > 1)
        uv.y -= 1;
    while (uv.y < 0)
        uv.y += 1;
    const int x = 2 * static_cast<int>(static_cast<int>(uv.x * 2) % 2) - 1,
              y = 2 * static_cast<int>(static_cast<int>(uv.y * 2) % 2) - 1;
    return (x * y == 1) ? color0_ : color1_;
}

QUALIFIER_DEVICE Vec2 CheckerboardTexture::GetGradient(const Vec2 &texcoord,
                                                       const float *pixel_buffer) const
{
    constexpr float delta = 1e-4f, norm = 1.0f / delta;
    const float value = Length(GetColor(texcoord, pixel_buffer)),
                value_u = Length(GetColor(Vec2{texcoord.u + delta, texcoord.v}, pixel_buffer)),
                value_v = Length(GetColor(Vec2{texcoord.u, texcoord.v + delta}, pixel_buffer));
    return {(value_u - value) * norm, (value_v - value) * norm};
}

QUALIFIER_DEVICE Vec3 Bitmap::GetColor(const Vec2 &texcoord, const float *pixel_buffer) const
{
    float x = texcoord.u * width_, y = texcoord.v * height_;
    while (x < 0)
        x += width_;
    while (x > width_ - 1)
        x -= width_;
    while (y < 0)
        y += height_;
    while (y > height_ - 1)
        y -= height_;

    const int x_lower = static_cast<int>(x),
              y_lower = static_cast<int>(y);
    const float t_x = x - x_lower,
                t_y = y - y_lower;
    const int x_upper = (t_x > 0.0f) ? x_lower + 1 : x_lower,
              y_upper = (t_y > 0.0f) ? y_lower + 1 : y_lower;

    uint64_t offset = offset_ + (x_lower + width_ * y_lower) * channel_;
    const Vec3 color_lower_lower = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                   : Vec3{pixel_buffer[offset], pixel_buffer[offset + 1],
                                                          pixel_buffer[offset + 2]};

    offset = offset_ + (x_lower + width_ * y_upper) * channel_;
    const Vec3 color_lower_upper = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                   : Vec3{pixel_buffer[offset], pixel_buffer[offset + 1],
                                                          pixel_buffer[offset + 2]};

    offset = offset_ + (x_upper + width_ * y_lower) * channel_;
    const Vec3 color_upper_lower = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                   : Vec3{pixel_buffer[offset], pixel_buffer[offset + 1],
                                                          pixel_buffer[offset + 2]};

    offset = offset_ + (x_upper + width_ * y_upper) * channel_;
    const Vec3 color_upper_upper = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                   : Vec3{pixel_buffer[offset], pixel_buffer[offset + 1],
                                                          pixel_buffer[offset + 2]};

    const Vec3 color_lower_lerp = Lerp(color_lower_lower, color_lower_upper, t_y),
               color_upper_lerp = Lerp(color_upper_lower, color_upper_upper, t_y);
    return Lerp(color_lower_lerp, color_upper_lerp, t_x);
}

QUALIFIER_DEVICE Vec2 Bitmap::GetGradient(const Vec2 &texcoord, const float *pixel_buffer) const
{
    auto GetNorm = [&](const Vec2 &texcoord, int offset_x, int offset_y) -> float
    {
        float x = texcoord.u * width_ + offset_x, y = texcoord.v * height_ + offset_y;
        while (x < 0)
            x += width_;
        while (x > width_ - 1)
            x -= width_;
        while (y < 0)
            y += height_;
        while (y > height_ - 1)
            y -= height_;

        const int x_lower = static_cast<int>(x),
                  y_lower = static_cast<int>(y);
        const float t_x = x - x_lower,
                    t_y = y - y_lower;
        const int x_upper = (t_x > 0.0f) ? x_lower + 1 : x_lower,
                  y_upper = (t_y > 0.0f) ? y_lower + 1 : y_lower;

        uint64_t offset = offset_ + (x_lower + width_ * y_lower) * channel_;
        const Vec3 color_lower_lower = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                       : Vec3{pixel_buffer[offset],
                                                              pixel_buffer[offset + 1],
                                                              pixel_buffer[offset + 2]};

        offset = offset_ + (x_lower + width_ * y_upper) * channel_;
        const Vec3 color_lower_upper = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                       : Vec3{pixel_buffer[offset],
                                                              pixel_buffer[offset + 1],
                                                              pixel_buffer[offset + 2]};

        offset = offset_ + (x_upper + width_ * y_lower) * channel_;
        const Vec3 color_upper_lower = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                       : Vec3{pixel_buffer[offset],
                                                              pixel_buffer[offset + 1],
                                                              pixel_buffer[offset + 2]};

        offset = offset_ + (x_upper + width_ * y_upper) * channel_;
        const Vec3 color_upper_upper = (channel_ == 1) ? Vec3(pixel_buffer[offset])
                                                       : Vec3{pixel_buffer[offset],
                                                              pixel_buffer[offset + 1],
                                                              pixel_buffer[offset + 2]};

        const Vec3 color_lower_lerp = Lerp(color_lower_lower, color_lower_upper, t_y),
                   color_upper_lerp = Lerp(color_upper_lower, color_upper_upper, t_y);
        if (channel_ == 1)
            return Lerp(color_lower_lerp, color_upper_lerp, t_x).x;
        else
            return Length(Lerp(color_lower_lerp, color_upper_lerp, t_x));
    };

    constexpr float kh = 2.1f, kn = 2.1f;
    const float value = GetNorm(texcoord, 0, 0),
                value_u = GetNorm(texcoord, 1, 0),
                value_v = GetNorm(texcoord, 0, 1);
    float du = kh * kn * (value_u - value),
          dv = kh * kn * (value_v - value);
    return {du, dv};
}

QUALIFIER_DEVICE bool Bitmap::IsTransparent(const Vec2 &texcoord, const float *pixel_buffer,
                                            uint64_t *seed) const
{
    if (channel_ != 1 && channel_ != 4)
        return false;

    float x = texcoord.u * width_, y = texcoord.v * height_;
    while (x < 0)
        x += width_;
    while (x > width_ - 1)
        x -= width_;
    while (y < 0)
        y += height_;
    while (y > height_ - 1)
        y -= height_;

    const int x_lower = static_cast<int>(x),
              y_lower = static_cast<int>(y);
    const float t_x = x - x_lower,
                t_y = y - y_lower;
    const int x_upper = (t_x > 0.0f) ? x_lower + 1 : x_lower,
              y_upper = (t_y > 0.0f) ? y_lower + 1 : y_lower;

    uint64_t offset = offset_ + (x_lower + width_ * y_lower) * channel_;
    const float aplha_lower_lower = (channel_ == 1) ? pixel_buffer[offset] : pixel_buffer[offset + 3];

    offset = offset_ + (x_lower + width_ * y_upper) * channel_;
    const float aplha_lower_upper = (channel_ == 1) ? pixel_buffer[offset] : pixel_buffer[offset + 3];

    offset = offset_ + (x_upper + width_ * y_lower) * channel_;
    const float aplha_upper_lower = (channel_ == 1) ? pixel_buffer[offset] : pixel_buffer[offset + 3];

    offset = offset_ + (x_upper + width_ * y_upper) * channel_;
    const float aplha_upper_upper = (channel_ == 1) ? pixel_buffer[offset] : pixel_buffer[offset + 3];

    const float aplha_lower_lerp = Lerp(aplha_lower_lower, aplha_lower_upper, t_y),
                aplha_upper_lerp = Lerp(aplha_upper_lower, aplha_upper_upper, t_y);

    const float alpha = Lerp(aplha_lower_lerp, aplha_upper_lerp, t_x);
    return RandomFloat(seed) > alpha;
}
