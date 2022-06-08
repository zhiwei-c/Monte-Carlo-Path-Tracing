#pragma once

#include <vector>
#include <string>
#include <set>

#include "../utils/global.h"
#include "../utils/file_path.h"
#include "../utils/image_io.h"

enum TextureType
{
    kNoneTexture, //空纹理
    kConstant,    //恒定颜色
    kBitmap,      //位图
};

struct TextureInfo
{
    TextureType type;
    int width;
    int height;
    int channel;
    Float line_width;
    std::vector<float> colors;
    vec3 color;
    std::string filename;

    TextureInfo(Float color) : type(kConstant), width(1), height(1), channel(3), color(vec3(color)) {}

    TextureInfo(const vec3 &color) : type(kConstant), width(1), height(1), channel(3), color(color) {}

    TextureInfo(const std::string &filename, Float gamma) : type(kBitmap), filename(filename)
    {
        ImageReader(filename, gamma, width, height, channel, colors);
    }
};

class Texture
{
public:
    __device__ Texture() : type_(kNoneTexture), width_(0), height_(0), channel_(0), color_(vec3(0)), colors_(nullptr) {}

    __device__ vec3 Color(const vec2 &texcoord) const;

    __device__ vec2 Gradient(const vec2 &texcoord) const;

    __device__ bool Transparent(const vec2 &texcoord, Float sample) const;

    __device__ bool IsBitmap() const { return type_ == kBitmap; }

    __device__ bool Varying() const { return type_ != kConstant; }

    __device__ void InitConstant(const vec3 &color);

    __device__ void InitBitmap(int width, int height, int channel, float *colors);

private:
    __device__ Float GetNorm(const vec2 &coord, int offset_x, int offset_y) const;

    TextureType type_;
    int width_;
    int height_;
    int channel_;
    vec3 color_;
    float *colors_;
};

__global__ inline void InitBitmapTexture(uint idx, int width, int height, int channel, float *colors, Texture *texture_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        texture_list[idx].InitBitmap(width, height, channel, colors);
    }
}

__global__ inline void InitConstantTexture(uint idx, vec3 color, Texture *texture_list)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        texture_list[idx].InitConstant(color);
    }
}
