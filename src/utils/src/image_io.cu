#include "image_io.cuh"

#include <algorithm>
#include <array>
#include <set>

#include <cmath>
#include <cstdio>
#include <tinyexr.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
extern "C"
{
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"
}

#include "misc.cuh"

namespace csrt
{

void image_io::Write(const int width, const int height,
                     const float *frame_buffer, const std::string &filename)
{
    unsigned char *color = new unsigned char[width * height * 3];
    int offset = 0;
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            offset = (j * width + i) * 3;
            color[offset] = static_cast<unsigned char>(
                std::min(255, static_cast<int>(255 * frame_buffer[offset])));
            color[offset + 1] = static_cast<unsigned char>(std::min(
                255, static_cast<int>(255 * frame_buffer[offset + 1])));
            color[offset + 2] = static_cast<unsigned char>(std::min(
                255, static_cast<int>(255 * frame_buffer[offset + 2])));
        }
    }

    int ret =
        stbi_write_png(filename.c_str(), width, height, 3, color, width * 3);
    delete[] color;
    color = nullptr;
    if (ret == 0)
    {
        fprintf(stderr, "[error] write image failed.\n");
    }
    else
    {
        fprintf(stderr, "[info] save result as image \"%s\".\n",
                filename.c_str());
    }
}

void image_io::Read(const std::string &filename, const float gamma,
                    const int *width_max, int *width, int *height, int *channel,
                    std::vector<float> *data)
{
    const std::set<std::string> support_format = {
        "EXR", "exr", "jpg", "jpeg", "JPG", "JPEG", "png", "PNG",
        "tga", "TGA", "bmp", "BMP",  "psd", "PSD",  "gif", "GIF",
        "hdr", "HDR", "pic", "PIC",  "pgm", "PGM",  "ppm", "PPM"};
    std::string suffix = GetSuffix(filename);
    if (!support_format.count(suffix))
    {
        fprintf(stderr, "[error] unsupport input image format for image '%s\n'",
                filename.c_str());
        exit(1);
    }

    float *raw_data = nullptr;
    switch (Hash(suffix.c_str()))
    {
    case "exr"_hash:
    case "EXR"_hash:
    {
        const char *err = nullptr;
        if (LoadEXR(&raw_data, width, height, filename.c_str(), &err) !=
            TINYEXR_SUCCESS)
        {
            fprintf(stderr, "[error] load image '%s' failed.",
                    filename.c_str());
            if (err)
            {
                fprintf(stderr, "\t%s", err);
                FreeEXRErrorMessage(err);
            }
            exit(1);
        }
        if (gamma != 0.0f)
        {
            int num_component = *width * *height * *channel;
            for (int i = 0; i < num_component; ++i)
                raw_data[i] = std::pow(raw_data[i], gamma);
        }
        *channel = 4;
        break;
    }
    default:
    {
        stbi_uc *raw_data_uc =
            stbi_load(filename.c_str(), width, height, channel, 0);
        if (raw_data_uc == nullptr)
        {
            fprintf(stderr, "[error] load image '%s' failed.",
                    filename.c_str());
            exit(1);
        }
        int num_component = *width * *height * *channel;
        raw_data = new float[num_component];
        if (suffix != "HDR" && suffix != "hdr")
        {
            for (int i = 0; i < num_component; ++i)
                raw_data[i] = static_cast<int>(raw_data_uc[i]) / 255.0f;
            if (gamma == 0.0f || gamma == -1.0f)
            {
                for (int i = 0; i < num_component; ++i)
                {
                    raw_data[i] =
                        raw_data[i] <= 0.04045f
                            ? raw_data[i] / 12.92f
                            : std::pow((raw_data[i] + 0.055f) / 1.055f, 2.4f);
                }
            }
            else
            {
                for (int i = 0; i < num_component; ++i)
                    raw_data[i] = std::pow(raw_data[i], gamma);
            }
        }
        else
        {
            for (int i = 0; i < num_component; ++i)
                raw_data[i] = static_cast<int>(raw_data_uc[i]);

            if (gamma == -1.0f)
            {
                for (int i = 0; i < num_component; ++i)
                {
                    raw_data[i] =
                        raw_data[i] <= 0.04045f
                            ? raw_data[i] / 12.92f
                            : std::pow((raw_data[i] + 0.055f) / 1.055f, 2.4f);
                }
            }
            else if (gamma != 0.0f)
            {
                for (int i = 0; i < num_component; ++i)
                    raw_data[i] = std::pow(raw_data[i], gamma);
            }
        }

        stbi_image_free(raw_data_uc);
        break;
    }
    }

    if (width_max != nullptr && *width > *width_max)
    {
        int height_target = *width_max * *height / *width;
        float *target_data = new float[*width_max * height_target * *channel];
        Resize(raw_data, *width, *height, 0, target_data, *width_max,
               height_target, 0, *channel);
        raw_data = target_data;
        *width = *width_max;
        *height = height_target;
    }

    *data =
        std::vector<float>(raw_data, raw_data + *width * *height * *channel);
    SAFE_DELETE_ARRAY(raw_data);
}

void image_io::Resize(const float *input_pixels, int input_w, int input_h,
                      int input_stride_in_bytes, float *output_pixels,
                      int output_w, int output_h, int output_stride_in_bytes,
                      int num_channels)
{
    stbir_resize_float_linear(input_pixels, input_w, input_h,
                              input_stride_in_bytes, output_pixels, output_w,
                              output_h, output_stride_in_bytes,
                              static_cast<stbir_pixel_layout>(num_channels));
}

} // namespace csrt