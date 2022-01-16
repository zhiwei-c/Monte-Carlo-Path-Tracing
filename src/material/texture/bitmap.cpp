#include "bitmap.h"

#include <iostream>
#include <algorithm>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "tinyexr.h"
#include "../../utils/file_path.h"
#include "../../utils/math/maths.h"

NAMESPACE_BEGIN(simple_renderer)

static Float UndoGamma(Float value, Float gamma)
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

static Float ApplyGamma(Float value, Float gamma_inv)
{
    if (gamma_inv == -1)
        return (value <= (Float)0.0031308) ? ((Float)12.92 * value)
                                           : ((Float)1.055 * std::pow(value, (Float)(1.0 / 2.4)) - (Float)0.055);
    else
        return std::pow(value, gamma_inv);
}

const std::set<std::string> kStbInputFormat = {"jpg", "jpeg", "JPG", "JPEG",
                                               "png", "PNG",
                                               "tga", "TGA",
                                               "bmp", "BMP",
                                               "psd", "PSD",
                                               "gif", "GIF",
                                               "hdr", "HDR",
                                               "pic", "PIC",
                                               "pgm", "PGM", "ppm", "PPM"};

Bitmap::Bitmap(const std::string &file_name, Float gamma)
    : Texture(TextureType::kBitmap), file_name_(file_name), gamma_(gamma), gamma_inv_(1.0 / gamma)
{
    auto suffix = GetSuffix(file_name);
    if (kStbInputFormat.find(suffix) != kStbInputFormat.end())
    {
        if (auto data = stbi_load(file_name.c_str(), &width_, &height_, &channels_, 0);
            data != nullptr)
        {
            auto cnt = static_cast<size_t>(width_) * height_ * channels_;
            data_.resize(cnt);
            if (gamma_ == 1)
                for (size_t i = 0; i < cnt; i++)
                    data_[i] = static_cast<float>(data[i]) / 255;
            else
                for (size_t i = 0; i < cnt; i++)
                    data_[i] = UndoGamma(static_cast<float>(data[i]) / 255, gamma_);
        }
        else
        {
            std::cerr << "[error] load image \"" << file_name << "\" failed." << std::endl;
            exit(1);
        }
    }
    else if (suffix == "exr")
    {
        float *data; // width * height * RGBA
        const char *err = nullptr;
        int ret = LoadEXR(&data, &width_, &height_, file_name.c_str(), &err);
        channels_ = 4;
        if (ret != TINYEXR_SUCCESS)
        {
            std::cerr << "[error] load image \"" << file_name << "\" failed." << std::endl;
            if (err)
            {
                std::cerr << "[error info] :" << err << std::endl;
                FreeEXRErrorMessage(err);
            }
            exit(1);
        }
        else
        {
            auto cnt = static_cast<size_t>(width_) * height_ * channels_;
            data_.resize(cnt);
            if (gamma_ == 1)
                for (size_t i = 0; i < cnt; i++)
                    data_[i] = data[i];
            else
                for (size_t i = 0; i < cnt; i++)
                    data_[i] = UndoGamma(data[i], gamma_);
            free(data);
        }
    }
    else
    {
        std::cerr << "[error] unsupport input image format \"" << suffix
                  << "\" for image:" << file_name << std::endl;
        exit(1);
    }
}

Bitmap::Bitmap(int width, int height, int channels, Float gamma)
    : Texture(TextureType::kBitmap),
      file_name_(""),
      width_(width),
      height_(height),
      channels_(channels),
      gamma_(gamma),
      gamma_inv_(1.0 / gamma)
{
    data_.resize(width * height * channels);
    std::fill(data_.begin(), data_.end(), 0);
}

void Bitmap::SetPixel(int x, int y, const Vector3 &value)
{
    auto offset = (static_cast<size_t>(x) + width_ * y) * channels_;
    data_[offset] = ClampBottom<Float>(0, value.r);
    data_[offset + 1] = ClampBottom<Float>(0, value.g);
    data_[offset + 2] = ClampBottom<Float>(0, value.b);
}

Spectrum Bitmap::GetPixel(const Vector2 &coord) const
{
    auto x = static_cast<int>(coord.x * width_),
         y = static_cast<int>(coord.y * height_);
    x = Modulo(x, width_);
    y = Modulo(y, height_);
    auto offset = (x + static_cast<size_t>(width_) * y) * channels_;
    auto r = data_[offset];
    auto g = data_[offset + 1];
    auto b = data_[offset + 2];
    return Spectrum(r, g, b);
}

Vector2 Bitmap::GetGradient(const Vector2 &coord) const
{
    auto GetNorm = [&](const Vector2 &coord, int offset_x, int offset_y)
    {
        auto x = static_cast<int>(coord.x * width_) + offset_x,
             y = static_cast<int>(coord.y * height_) + offset_y;
        x = Modulo(x, width_);
        y = Modulo(y, height_);
        auto offset = (x + static_cast<size_t>(width_) * y) * channels_;
        auto r = 255 * data_[offset];
        auto g = 255 * data_[offset + 1];
        auto b = 255 * data_[offset + 2];
        return glm::length(Vector3(r, g, b));
    };
    Float kh = 0.2, kn = 0.1;
    auto value = GetNorm(coord, 0, 0),
         value_u = GetNorm(coord, 1, 0),
         value_v = GetNorm(coord, 0, 1);
    auto du = kh * kn * (value_u - value),
         dv = kh * kn * (value_v - value);
    return Vector2(du, dv);
}

void Bitmap::Write(const std::string &path)
{
    int ret = 0;

    auto suffix = GetSuffix(path);
    if (suffix == "exr" || suffix == "EXR")
    {
        ret = WriteOpenexr(path);
        return;
    }
    else if (suffix == "hdr" || suffix == "HDR")
    {
        auto data = new float[width_ * height_ * channels_];
        if (gamma_ == 1)
            for (auto i = 0; i < data_.size(); i++)
                data[i] = data_[i];
        else
            for (auto i = 0; i < data_.size(); i++)
                data[i] = ApplyGamma(data_[i], gamma_inv_);

        ret = stbi_write_hdr(path.c_str(), width_, height_, channels_, data);
        stbi_image_free(data);
        data = nullptr;
    }
    else
    {
        auto data = new unsigned char[width_ * height_ * channels_];
        if (gamma_ == 1)
            for (auto i = 0; i < data_.size(); i++)
                data[i] = static_cast<unsigned char>(ClampTop<int>(255, 255 * data_[i]));
        else
            for (auto i = 0; i < data_.size(); i++)
                data[i] = static_cast<unsigned char>(ClampTop<int>(255, 255 * ApplyGamma(data_[i], gamma_inv_)));

        switch (Hash(suffix.c_str()))
        {
        case "JPG"_hash:
        case "JPEG"_hash:
        case "jpg"_hash:
        case "jpeg"_hash:
            ret = stbi_write_jpg(path.c_str(), width_, height_, channels_, data, 95);
            break;
        case "PNG"_hash:
        case "png"_hash:
            ret = stbi_write_png(path.c_str(), width_, height_, channels_, data, width_ * channels_);
            break;
        default:
            std::cout << "[warning] unsupported output format \"" << suffix << "\", use png instead." << std::endl;
            ret = stbi_write_png(ChangeSuffix(path, "png").c_str(), width_, height_, channels_, data, width_ * channels_);
            break;
        }
        stbi_image_free(data);
        data = nullptr;
    }

    if (ret == 0)
    {
        std::cout << "[error]" << path << std::endl
                  << "\twrite image failed." << std::endl;
        exit(1);
    }
}

bool Bitmap::Transparent(const Vector2 &coord) const
{
    if (channels_ != 4)
        return false;

    auto x = static_cast<int>(coord.x * width_),
         y = static_cast<int>(coord.y * height_);
    x = Modulo(x, width_);
    y = Modulo(y, height_);
    auto offset = (static_cast<size_t>(x) + width_ * y) * channels_;
    auto alpha = data_[offset + 3];
    if (alpha == 1)
        return false;
    else if (alpha == 0)
        return true;
    else
    {
        auto xi = UniformFloat();
        if (xi < alpha)
            return false;
        else
            return true;
    }
}

int Bitmap::WriteOpenexr(const std::string &path)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    auto resolution_ = static_cast<size_t>(width_) * height_;

    images[0].resize(resolution_);
    images[1].resize(resolution_);
    images[2].resize(resolution_);

    for (size_t i = 0; i < resolution_; i++)
    {
        images[0][i] = data_[3 * i + 0];
        images[1][i] = data_[3 * i + 1];
        images[2][i] = data_[3 * i + 2];
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char **)image_ptr;
    image.width = width_;
    image.height = height_;

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
NAMESPACE_END(simple_renderer)