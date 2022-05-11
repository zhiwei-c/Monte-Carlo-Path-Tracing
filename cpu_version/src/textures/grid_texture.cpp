#include "grid_texture.h"

NAMESPACE_BEGIN(raytracer)

///\brief 过程式网格纹理
GridTexture::GridTexture(const Spectrum &color0, const Spectrum &color1, Float line_width,
                         const Vector2 &uv_offset, const Vector2 &uv_scale)
    : Texture(TextureType::kCheckerboard),
      color0_(color0), color1_(color1), line_width_(line_width),
      uv_offset_(nullptr), uv_scale_(nullptr)
{
    if (uv_offset.x > kEpsilon || uv_offset.y > kEpsilon)
        uv_offset_ = std::make_unique<Vector2>(uv_offset);
    if (!FloatEqual(uv_scale.x, 1) || !FloatEqual(uv_scale.y, 1))
        uv_scale_ = std::make_unique<Vector2>(uv_scale);
}

///\return 纹理在给定坐标处像素值
Spectrum GridTexture::Color(const Vector2 &coord) const
{
    auto u = coord.x, v = coord.y;
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

    Float x = u - static_cast<int>(std::floor(u)),
          y = v - static_cast<int>(std::floor(v));

    if (x > .5)
        x -= 1;
    if (y > .5)
        y -= 1;

    if (std::abs(x) < line_width_ || std::abs(y) < line_width_)
        return color1_;
    else
        return color0_;
}

///\return 纹理在给定坐标处梯度
Vector2 GridTexture::Gradient(const Vector2 &coord) const
{
    Float value = glm::length(Color(coord)),
          value_u = glm::length(Color({coord.x + 1e-4, coord.y})),
          value_v = glm::length(Color({coord.x, coord.y + 1e-4}));
    Float du = (value_u - value) * (1 / 1e-4),
          dv = (value_v - value) * (1 / 1e-4);
    return Vector2(du, dv);
}

NAMESPACE_END(raytracer)