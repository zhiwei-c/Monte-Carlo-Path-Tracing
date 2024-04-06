#include "checkerboard.hpp"

#include "../math/coordinate.hpp"

NAMESPACE_BEGIN(raytracer)

Checkerboard::Checkerboard(const std::string &id, Texture *color0, Texture *color1, dmat3 to_uv)
    : Texture(TextureType::kCheckerboard, id),
      color0_(color0),
      color1_(color1),
      to_uv_(to_uv)
{
}
    
dvec3 Checkerboard::color(const dvec2 &texcoord) const
{
    dvec2 uv = TransfromPoint(to_uv_, texcoord);
    while (uv.x > 1)
    {
        uv.x -= 1;
    }
    while (uv.x < 0)
    {
        uv.x += 1;
    }
    while (uv.y > 1)
    {
        uv.y -= 1;
    }
    while (uv.y < 0)
    {
        uv.y += 1;
    }

    int x = 2 * static_cast<int>(static_cast<int>(uv.x * 2) % 2) - 1,
        y = 2 * static_cast<int>(static_cast<int>(uv.y * 2) % 2) - 1;

    if (x * y == 1)
    {
        return color0_->color(uv);
    }
    else
    {
        return color1_->color(uv);
    }
}

dvec2 Checkerboard::gradient(const dvec2 &texcoord) const
{
    const double value = glm::length(color(texcoord)),
                 value_u = glm::length(color(dvec2{texcoord.x + 1e-4, texcoord.y})),
                 value_v = glm::length(color(dvec2{texcoord.x, texcoord.y + 1e-4}));
    constexpr double norm = 1.0 / 1e-4;
    return {(value_u - value) * norm, (value_v - value) * norm};
}

NAMESPACE_END(raytracer)