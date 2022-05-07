#pragma once

#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "tinyexr.h"

#include "file_path.h"
#include "global.h"

struct Frame
{
    int width;
    int height;
    std::vector<float> data;

    Frame() : width(0), height(0) {}
};

inline int WriteOpenexr(Frame &frame, const std::string &path)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    auto resolution = frame.width * frame.height;

    images[0].resize(resolution);
    images[1].resize(resolution);
    images[2].resize(resolution);

    for (size_t i = 0; i < resolution; i++)
    {
        images[0][i] = frame.data[3 * i + 0];
        images[1][i] = frame.data[3 * i + 1];
        images[2][i] = frame.data[3 * i + 2];
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char **)image_ptr;
    image.width = frame.width;
    image.height = frame.height;

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
    int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);

    if (ret != TINYEXR_SUCCESS)
    {
        fprintf(stderr, "Save EXR err: %s\n", err);
        return ret;
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    return ret;
}

inline void WriteImage(Frame &frame, const std::string &path)
{
    auto ret = static_cast<int>(0);

    auto suffix = GetSuffix(path);

    if (suffix == "exr" || suffix == "EXR")
    {
        ret = WriteOpenexr(frame, path);
        return;
    }

    if (suffix == "hdr" || suffix == "HDR")
    {
        ret = stbi_write_hdr(path.c_str(), frame.width, frame.height, 3, frame.data.data());
        return;
    }

    auto data = new unsigned char[frame.width * frame.height * 3];

    for (auto i = 0; i < frame.data.size(); i++)
        data[i] = static_cast<unsigned char>(std::min(255, static_cast<int>(255 * frame.data[i])));

    switch (Hash(suffix.c_str()))
    {
    case "JPG"_hash:
    case "JPEG"_hash:
    case "jpg"_hash:
    case "jpeg"_hash:
        ret = stbi_write_jpg(path.c_str(), frame.width, frame.height, 3, data, 95);
        break;
    case "PNG"_hash:
    case "png"_hash:
        ret = stbi_write_png(path.c_str(), frame.width, frame.height, 3, data, frame.width * 3);
        break;
    default:
        std::cout << "[warning] unsupported output format \"" << suffix << "\", use png instead." << std::endl;
        ret = stbi_write_png(ChangeSuffix(path, "png").c_str(), frame.width, frame.height, 3, data, frame.width * 3);
        break;
    }
    stbi_image_free(data);
    data = nullptr;

    if (ret == 0)
    {
        std::cout << "[error]" << path << std::endl
                  << "\twrite image failed." << std::endl;
        exit(1);
    }
}