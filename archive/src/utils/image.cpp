#include "image.hpp"

#include <iostream>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <tinyexr.h>

#include "file_path.hpp"
#include "../textures/bitmap.hpp"

NAMESPACE_BEGIN(raytracer)

void SaveSdrImage(const std::vector<float> &data, int width, int height, const std::string &filename);
void SaveExrImage(const std::vector<float> &data, int width, int height, const std::string &filename);

Texture *LoadImage(const std::string &filename, const std::string &id, int *max_width)
{
    const std::set<std::string> support_format = {"EXR", "exr",
                                                  "jpg", "jpeg", "JPG", "JPEG",
                                                  "png", "PNG",
                                                  "tga", "TGA",
                                                  "bmp", "BMP",
                                                  "psd", "PSD",
                                                  "gif", "GIF",
                                                  "hdr", "HDR",
                                                  "pic", "PIC",
                                                  "pgm", "PGM", "ppm", "PPM"};
    std::string suffix = GetSuffix(filename);
    if (!support_format.count(suffix))
    {
        std::cerr << "[error] unsupport input image format \"" << suffix
                  << "\" for image:" << filename << std::endl;
        exit(1);
    }

    auto data = std::vector<float>();
    int width = 0, height = 0, channels = 0;
    switch (Hash(suffix.c_str()))
    {
    case "exr"_hash:
    case "EXR"_hash:
    {
        float *raw_data = nullptr;
        const char *err = nullptr;
        if (LoadEXR(&raw_data, &width, &height, filename.c_str(), &err) != TINYEXR_SUCCESS)
        {
            std::cerr << "[error] load image \"" << filename << "\" failed." << std::endl;
            if (err)
            {
                std::cerr << "[error info] :" << err << std::endl;
                FreeEXRErrorMessage(err);
            }
            exit(1);
        }
        channels = 4;
        data = std::vector<float>(raw_data, raw_data + width * height * channels);
        free(raw_data);
        break;
    }
    default:
    {
        stbi_uc *raw_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        if (raw_data == nullptr)
        {
            std::cerr << "[error] load image \"" << filename << "\" failed." << std::endl;
            exit(1);
        }
        int data_size = width * height * channels;
        data.resize(data_size);
        for (int i = 0; i < data_size; ++i)
        {
            const float value = static_cast<int>(raw_data[i]) / 255.0f;
            data[i] = value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
        }
        stbi_image_free(raw_data);
        break;
    }
    }

    if (max_width != nullptr && width > *max_width)
    {
        int target_height = *max_width * height / width;
        auto final_data = std::vector<float>(*max_width * target_height * channels, 0);
        stbir_resize_float(data.data(), width, height, 0, final_data.data(), *max_width, target_height, 0, channels);
        data = final_data;
        width = *max_width;
        height = target_height;
    }
    return new Bitmap(id.empty() ? filename : id, data, width, height, channels);
}

void SaveImage(const std::vector<float> &data, int width, int height, const std::string &filename)
{
    std::string suffix = GetSuffix(filename);
    switch (Hash(suffix.c_str()))
    {
    case "exr"_hash:
    case "EXR"_hash:
        SaveExrImage(data, width, height, filename);
        break;
    case "hdr"_hash:
    case "HDR"_hash:
    {
        stbi_write_hdr(filename.c_str(), width, height, 3, data.data());
        break;
    }
    default:
        SaveSdrImage(data, width, height, filename);
        break;
    }
}

void SaveSdrImage(const std::vector<float> &data, int width, int height, const std::string &filename)
{
    auto raw_data = std::vector<unsigned char>(width * height * 3);
    for (size_t i = 0; i < data.size(); ++i)
    {
        float srgb_value = data[i] <= 0.0031308f ? (12.92f * data[i]) : (1.055f * std::pow(data[i], 1.0f / 2.4f) - 0.055f);
        raw_data[i] = static_cast<unsigned char>(static_cast<int>(srgb_value > 1.0f ? 255.0f : srgb_value * 255.0f));
    }

    auto suffix = GetSuffix(filename);
    switch (Hash(suffix.c_str()))
    {
    case "JPG"_hash:
    case "JPEG"_hash:
    case "jpg"_hash:
    case "jpeg"_hash:
        stbi_write_jpg(filename.c_str(), width, height, 3, raw_data.data(), 95);
        break;
    case "PNG"_hash:
    case "png"_hash:
        stbi_write_png(filename.c_str(), width, height, 3, raw_data.data(), width * 3);
        break;
    default:
        std::cerr << "[warning] unsupported output format \"" << suffix << "\", use png instead." << std::endl;
        stbi_write_png(ChangeSuffix(filename, "png").c_str(), width, height, 3, raw_data.data(), width * 3);
        break;
    }
}

void SaveExrImage(const std::vector<float> &data, int width, int height, const std::string &filename)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    int resolution = width * height;

    images[0].resize(resolution);
    images[1].resize(resolution);
    images[2].resize(resolution);
    for (int i = 0; i < resolution; i++)
    {
        images[0][i] = static_cast<float>(data[3 * i + 0]);
        images[1][i] = static_cast<float>(data[3 * i + 1]);
        images[2][i] = static_cast<float>(data[3 * i + 2]);
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char **)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be BGR(A) order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255);
    header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255);
    header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255);
    header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++)
    {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;          // pixel type of input image
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char *err;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);

    if (ret != TINYEXR_SUCCESS)
    {
        fprintf(stderr, "Save EXR err: %s\n", err);
        return;
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
}

NAMESPACE_END(raytracer)