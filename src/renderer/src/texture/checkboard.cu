#include "texture.cuh"

namespace rt
{

QUALIFIER_D_H Vec3 Texture::GetColorCheckerboard(const Vec2 &texcoord) const
{
    Vec3 uv = TransformPoint(data_.checkerboard.to_uv, {texcoord, 0.0f});
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
    return (x * y == 1) ? data_.checkerboard.color0 : data_.checkerboard.color1;
}

QUALIFIER_D_H Vec2 Texture::GetGradientCheckerboard(const Vec2 &texcoord) const
{
    constexpr float delta = 1e-4f, norm = 1.0f / delta;
    const float value = Length(GetColorCheckerboard(texcoord)),
                value_u =
                    Length(GetColorCheckerboard(texcoord + Vec2{delta, 0})),
                value_v =
                    Length(GetColorCheckerboard(texcoord + Vec2{0, delta}));
    return {(value_u - value) * norm, (value_v - value) * norm};
}

} // namespace rt