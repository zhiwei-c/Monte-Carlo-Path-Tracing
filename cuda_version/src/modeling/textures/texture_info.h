#pragma once

#include <vector>
#include <string>
#include <set>

#include "../../utils/global.h"
#include "../../utils/file_path.h"
#include "../../utils/image_writer.h"

enum TextureType
{
    kNoneTexture, //空纹理
    kConstant,    //恒定颜色
    kBitmap,
    kCheckerboard,
    kGridTexture
};

static const std::set<std::string> kStbInputFormat = {"jpg", "jpeg", "JPG", "JPEG",
                                                      "png", "PNG",
                                                      "tga", "TGA",
                                                      "bmp", "BMP",
                                                      "psd", "PSD",
                                                      "gif", "GIF",
                                                      "hdr", "HDR",
                                                      "pic", "PIC",
                                                      "pgm", "PGM",
                                                      "ppm", "PPM"};

inline static Float UndoGamma(Float value, Float gamma)
{
    if (gamma == -1)
    {
        if (value <= (Float)0.04045)
            return value * (Float)(1.0 / 12.92);
        else
            return std::pow((Float)((value + (Float)0.055) * (Float)(1.0 / 1.055)), (Float)2.4);
    }
    else
        return std::pow(value, gamma);
}

__device__ inline Float ApplyGamma(Float value, Float gamma_inv)
{
    if (gamma_inv == -1)
        return (value <= (Float)0.0031308) ? ((Float)12.92 * value)
                                           : ((Float)1.055 * pow(value, (Float)(1.0 / 2.4)) - (Float)0.055);
    else
        return pow(value, gamma_inv);
}


struct TextureInfo
{
    TextureType type;
    int width;
    int height;
    int channel;
    vec2 *uv_offset;
    vec2 *uv_scale;
    Float line_width;
    vec3 color0;
    vec3 color1;
    std::vector<float> colors;
    std::string filename;

    TextureInfo(Float color)
        : type(kConstant), width(1), height(1), channel(3)
    {
        for (int i = 0; i < 3; i++)
            colors.emplace_back(color);
    }

    TextureInfo(const vec3 &color)
        : type(kConstant), width(1), height(1), channel(3)
    {
        for (int i = 0; i < 3; i++)
            colors.emplace_back(color[i]);
    }

    TextureInfo(const vec3 &color0,
                const vec3 &color1,
                vec2 *uv_offset,
                vec2 *uv_scale)
        : type(kCheckerboard),
          line_width(line_width),
          uv_offset(uv_offset),
          uv_scale(uv_scale),
          color0(color0),
          color1(color1) {}

    TextureInfo(const vec3 &color0,
                const vec3 &color1,
                Float line_width,
                vec2 *uv_offset,
                vec2 *uv_scale)
        : type(kCheckerboard),
          line_width(line_width),
          uv_offset(uv_offset),
          uv_scale(uv_scale),
          color0(color0),
          color1(color1) {}

    TextureInfo(const std::string &filename, Float gamma)
        : type(kBitmap), filename(filename)
    {
        auto suffix = GetSuffix(filename);
        if (kStbInputFormat.find(suffix) != kStbInputFormat.end())
        {
            if (auto data = stbi_load(filename.c_str(), &width, &height, &channel, 0);
                data != nullptr)
            {
                auto cnt = static_cast<size_t>(width) * height * channel;
                colors.resize(cnt);
                if (gamma == 1)
                    for (size_t i = 0; i < cnt; i++)
                        colors[i] = static_cast<float>(data[i]) / 255.0;
                else
                    for (size_t i = 0; i < cnt; i++)
                        colors[i] = UndoGamma(static_cast<float>(data[i]) / 255.0, gamma);
            }
            else
            {
                std::cerr << "[error] load image \"" << filename << "\" failed." << std::endl;
                exit(1);
            }
        }
        else if (suffix == "exr")
        {
            float *data; // width * height * RGBA
            const char *err = nullptr;
            int ret = LoadEXR(&data, &width, &height, filename.c_str(), &err);
            channel = 4;
            if (ret != TINYEXR_SUCCESS)
            {
                std::cerr << "[error] load image \"" << filename << "\" failed." << std::endl;
                if (err)
                {
                    std::cerr << "[error info] :" << err << std::endl;
                    FreeEXRErrorMessage(err);
                }
                exit(1);
            }
            else
            {
                auto cnt = static_cast<size_t>(width) * height * channel;
                colors.resize(cnt);
                if (gamma == 1)
                    for (size_t i = 0; i < cnt; i++)
                        colors[i] = data[i];
                else
                    for (size_t i = 0; i < cnt; i++)
                        colors[i] = UndoGamma(data[i], gamma);
                free(data);
            }
        }
        else
        {
            std::cerr << "[error] unsupport input image format \"" << suffix
                      << "\" for image:" << filename << std::endl;
            exit(1);
        }
    }
};