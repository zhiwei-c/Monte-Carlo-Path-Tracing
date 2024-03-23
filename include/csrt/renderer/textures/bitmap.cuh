#pragma once

#include "../../tensor.cuh"
#include "../../utils.cuh"

namespace csrt
{

struct BitmapData
{
    int width = 0;
    int height = 0;
    int channel = 0;
    float *data = nullptr;
    Mat4 to_uv = {};
};

QUALIFIER_D_H Vec3 GetColorBitmap(const BitmapData &data, const Vec2 &texcoord);

QUALIFIER_D_H Vec2 GetGradientBitmap(const BitmapData &data,
                                     const Vec2 &texcoord);

QUALIFIER_D_H bool IsTransparentBitmap(const BitmapData &data,
                                       const Vec2 &texcoord, uint32_t *seed);

} // namespace csrt