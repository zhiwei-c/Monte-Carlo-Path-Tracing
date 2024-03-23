#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"

namespace csrt
{

struct CheckerboardData
{
    Vec3 color0 = {0.4f};
    Vec3 color1 = {0.2f};
    Mat4 to_uv = {};
};

QUALIFIER_D_H Vec3 GetColorCheckerboard(const CheckerboardData &data,
                                        const Vec2 &texcoord);

QUALIFIER_D_H Vec2 GetGradientCheckerboard(const CheckerboardData &data,
                                           const Vec2 &texcoord);

QUALIFIER_D_H bool IsTransparentCheckerboard(const CheckerboardData &data,
                                             const Vec2 &texcoord,
                                             uint32_t *seed);

} // namespace csrt