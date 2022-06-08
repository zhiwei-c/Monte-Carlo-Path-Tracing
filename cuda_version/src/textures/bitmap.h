#pragma once

#include "../core/texture_base.h"

__device__ void Texture::InitBitmap(int width, int height, int channel, float *colors)
{
    type_ = kBitmap;
    width_ = width;
    height_ = height;
    channel_ = channel;
    colors_ = colors;
}

__device__ vec3 Texture::ColorBitmap(const vec2 &texcoord) const
{
    auto x = static_cast<int>(texcoord.x * width_),
         y = static_cast<int>(texcoord.y * height_);
    x = Modulo(x, width_);
    y = Modulo(y, height_);
    auto offset = (x + static_cast<uint>(width_) * y) * channel_;
    auto r = colors_[offset];
    auto g = colors_[offset + 1];
    auto b = colors_[offset + 2];
    return vec3(r, g, b);
}

__device__ vec2 Texture::GradientBitmap(const vec2 &texcoord) const
{
    Float kh = 0.2, kn = 0.1;
    auto value = GetNorm(texcoord, 0, 0),
         value_u = GetNorm(texcoord, 1, 0),
         value_v = GetNorm(texcoord, 0, 1);
    auto du = kh * kn * (value_u - value),
         dv = kh * kn * (value_v - value);
    return vec2(du, dv);
}

__global__ void InitBitmapTexture(uint idx, int width, int height, int channel, float *colors, Texture *texture_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        texture_list[idx].InitBitmap(width, height, channel, colors);
    }
}