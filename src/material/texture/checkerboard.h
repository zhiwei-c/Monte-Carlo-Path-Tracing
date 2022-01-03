#pragma once

#include <string>

#include "../texture.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

//棋盘图
class Checkerboard : public Texture
{
public:
    //棋盘图
    Checkerboard(const Vector3 &color0, const Vector3 &color1, const Vector2 &uv_offset, const Vector2 &uv_scale)
        : Texture(TextureType::kCheckerboard),
          color0_(color0),
          color1_(color1),
          uv_offset_(nullptr),
          uv_scale_(nullptr)
    {
        if (uv_offset.x > kEpsilon || uv_offset.y > kEpsilon)
            uv_offset_ = std::make_unique<Vector2>(uv_offset);
        if (!FloatEqual(uv_scale.x, 1) || !FloatEqual(uv_scale.y, 1))
            uv_scale_ = std::make_unique<Vector2>(uv_scale);
    }

    Vector3 GetPixel(const Vector2 &coord) const override
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

        auto x = 2 * (int)((int)(u * 2) % 2) - 1;
        auto y = 2 * (int)((int)(v * 2) % 2) - 1;

        if (x * y == 1)
            return color0_;
        else
            return color1_;
    }

    Vector2 GetGradient(const Vector2 &coord) const override
    {
        auto value = glm::length(GetPixel(coord)),
             value_u = glm::length(GetPixel({coord.x + 1e-4, coord.y})),
             value_v = glm::length(GetPixel({coord.x, coord.y + 1e-4}));
        auto du = (value_u - value) * (1 / 1e-4),
             dv = (value_v - value) * (1 / 1e-4);
        return Vector2(du, dv);
    }

private:
    Vector3 color0_;
    Vector3 color1_;
    std::unique_ptr<Vector2> uv_offset_;
    std::unique_ptr<Vector2> uv_scale_;
};

NAMESPACE_END(simple_renderer)