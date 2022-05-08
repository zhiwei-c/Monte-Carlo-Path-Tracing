#pragma once

#include "../core/texture_base.h"

__device__ void Texture::InitGridTexture(const vec3 &color0,
                                         const vec3 &color1,
                                         Float line_width,
                                         vec2 *uv_scale,
                                         vec2 *uv_offset)
{
    type_ = kGridTexture;
    color0_ = color0;
    color1_ = color1;
    line_width_ = line_width;

    if (uv_scale)
        uv_scale_ = new vec2(*uv_scale);

    if (uv_offset)
        uv_offset_ = new vec2(*uv_offset);
}

__device__ vec3 Texture::ColorGridTexture(const vec2 &texcoord) const
{
    auto u = texcoord.x, v = texcoord.y;
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

    auto x = u - static_cast<int>(u),
         y = v - static_cast<int>(v);

    if (x > 0.5)
        x -= 1;
    if (y > 0.5)
        y -= 1;

    if (abs(x) < line_width_ || abs(y) < line_width_)
        return color1_;
    else
        return color0_;
}

__device__ vec2 Texture::GradientGridTexture(const vec2 &texcoord) const
{
    auto value = ColorGridTexture(texcoord).length(),
         value_u = ColorGridTexture({texcoord.x + 1e-4, texcoord.y}).length(),
         value_v = ColorGridTexture({texcoord.x, texcoord.y + 1e-4}).length();
    auto du = (value_u - value) * (1.0 / 1e-4),
         dv = (value_v - value) * (1.0 / 1e-4);
    return vec2(du, dv);
}

__global__ void InitGridTexture(uint idx,
                                vec3 color0,
                                vec3 color1,
                                Float line_width,
                                vec2 *uv_scale,
                                vec2 *uv_offset,
                                Texture *texture_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        texture_list[idx].InitGridTexture(color0,
                                          color1,
                                          line_width,
                                          uv_scale,
                                          uv_offset);
    }
}

void inline CreateGridTexture(uint texture_idx,
                              const std::vector<TextureInfo *> &texture_info_list,
                              Texture *&texture_list_)
{

    auto uv_scale = static_cast<vec2 *>(nullptr);
    if (texture_info_list[texture_idx]->uv_scale)
    {
        CheckCudaErrors(cudaMallocManaged(&uv_scale, sizeof(vec2)));
        cudaMemcpy(uv_scale, texture_info_list[texture_idx]->uv_scale, sizeof(vec2), cudaMemcpyHostToDevice);
    }
    auto uv_offset = static_cast<vec2 *>(nullptr);
    if (texture_info_list[texture_idx]->uv_offset)
    {
        CheckCudaErrors(cudaMallocManaged(&uv_offset, sizeof(vec2)));
        cudaMemcpy(uv_offset, texture_info_list[texture_idx]->uv_offset, sizeof(vec2), cudaMemcpyHostToDevice);
    }
    InitGridTexture<<<1, 1>>>(texture_idx,
                              texture_info_list[texture_idx]->color0,
                              texture_info_list[texture_idx]->color1,
                              texture_info_list[texture_idx]->line_width,
                              uv_scale,
                              uv_offset,
                              texture_list_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(uv_scale));
    CheckCudaErrors(cudaFree(uv_offset));
    uv_scale = nullptr;
    uv_offset = nullptr;
}