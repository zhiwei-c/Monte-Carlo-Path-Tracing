#pragma once

#include "texture_info.h"

__device__ inline int Modulo(int a, int b)
{
    auto c = a % b;
    if (c < 0)
        c += b;
    return c;
}

__device__ inline Float CyclicClamp(Float num)
{
    while (num > 1)
        num -= 1;
    while (num < 0)
        num += 1;
    return num;
}

class Texture
{
public:
    __device__ Texture()
        : type_(kNoneTexture),
          width_(0),
          height_(0),
          channel_(0),
          color_(vec3(0)),
          colors_(nullptr) {}

    __device__ vec3 Color(const vec2 &texcoord) const;

    __device__ vec2 Gradient(const vec2 &texcoord) const;

    __device__ bool Transparent(const vec2 &texcoord, Float sample) const;

    __device__ TextureType type() const { return type_; }

    __device__ void InitConstant(const vec3 &color);
    __device__ void InitBitmap(int width, int height, int channel, float *colors);

private:
    TextureType type_;
    int width_;
    int height_;
    int channel_;
    vec3 color_;
    float *colors_;

    __device__ Float GetNorm(const vec2 &coord, int offset_x, int offset_y) const;

    __device__ vec3 ColorBitmap(const vec2 &texcoord) const;
    __device__ vec2 GradientBitmap(const vec2 &texcoord) const;
};