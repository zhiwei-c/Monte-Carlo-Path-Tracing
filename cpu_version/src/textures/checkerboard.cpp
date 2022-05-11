#include "checkerboard.h"

NAMESPACE_BEGIN(raytracer)

//棋盘图
Checkerboard::Checkerboard(const Spectrum &color0, const Spectrum &color1,
                           const Vector2 &uv_offset, const Vector2 &uv_scale)
    : Texture(TextureType::kCheckerboard),
      color0_(color0), color1_(color1), uv_offset_(nullptr), uv_scale_(nullptr)
{
    if (uv_offset.x > kEpsilon || uv_offset.y > kEpsilon)
        uv_offset_ = std::make_unique<Vector2>(uv_offset);
    if (!FloatEqual(uv_scale.x, 1) || !FloatEqual(uv_scale.y, 1))
        uv_scale_ = std::make_unique<Vector2>(uv_scale);
}

///\return 纹理在给定坐标处像素值
Spectrum Checkerboard::Color(const Vector2 &coord) const
{
    Float u = coord.x, v = coord.y;
    if (uv_scale_)
    {
        u *= (*uv_scale_).x;
        v *= (*uv_scale_).y;
    }
    if (uv_offset_)
    {
        u += (*uv_offset_).x;
        v += (*uv_offset_).y;
    }
    u = CyclicClamp(u);
    v = CyclicClamp(v);

    int x = 2 * (int)((int)(u * 2) % 2) - 1,
        y = 2 * (int)((int)(v * 2) % 2) - 1;

    if (x * y == 1)
        return color0_;
    else
        return color1_;
}

///\return 纹理在给定坐标处梯度
Vector2 Checkerboard::Gradient(const Vector2 &coord) const
{
    Float value = glm::length(Color(coord)),
          value_u = glm::length(Color({coord.x + 1e-4, coord.y})),
          value_v = glm::length(Color({coord.x, coord.y + 1e-4}));
    Float du = (value_u - value) * (1.0 / 1e-4),
          dv = (value_v - value) * (1.0 / 1e-4);
    return Vector2(du, dv);
}

NAMESPACE_END(raytracer)