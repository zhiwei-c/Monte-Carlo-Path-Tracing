#pragma once

#include "../core/texture_base.h"

__device__ void Texture::InitConstant(const vec3 &color)
{
    type_ = kConstant;
    width_ = 1;
    height_ = 1;
    channel_ = 3;
    color_ = color;
}

__global__ void InitConstantTexture(uint idx, vec3 color, Texture *texture_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        texture_list[idx].InitConstant(color);
    }
}
