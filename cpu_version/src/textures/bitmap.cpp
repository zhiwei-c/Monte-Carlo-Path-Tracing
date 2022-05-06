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

#include "../utils/file_path.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 撤销伽马校正
Float UndoGamma(Float value, Float gamma);

///\brief 应用伽马校正
Float ApplyGamma(Float value, Float gamma_inv);

///\brief 通过 tinyexr 加载 HDR 图像
void LoadByTinyExr(const std::string &filename, Float gamma, std::vector<Float> &data, int &width, int &height, int &channels);

///\brief 通过 stb 加载图像
void LoadByStbImage(const std::string &filename, Float gamma, std::vector<Float> &data, int &width, int &height, int &channels);

///\brief 通过 stb 保存 SDR 图像
void SaveSdrByStbImage(const std::string &filename, Float gamma_inv, const std::vector<Float> &data, int width, int height);

///\brief 通过 tinyexr 保存 HDR 图像
void SaveHdrByTinyExr(const std::string &filename, const std::vector<Float> &data, int width, int height);

//支持的图像格式
const std::set<std::string> kStbInputFormat = {"EXR", "exr",
                                               "jpg", "jpeg", "JPG", "JPEG",
                                               "png", "PNG",
                                               "tga", "TGA",
                                               "bmp", "BMP",
                                               "psd", "PSD",
                                               "gif", "GIF",
                                               "hdr", "HDR",
                                               "pic", "PIC",
                                               "pgm", "PGM", "ppm", "PPM"};

///\brief 位图
Bitmap::Bitmap(int width, int height, int channels, Float gamma)
    : Texture(TextureType::kBitmap),
      width_(width),
      height_(height),
      channels_(channels),
      gamma_(gamma),
      gamma_inv_(1.0 / gamma)
{
    data_.resize(width * height * channels);
    std::fill(data_.begin(), data_.end(), 0);
}

///\brief 空白位图
Bitmap::Bitmap(const std::string &filename, Float gamma)
    : Texture(TextureType::kBitmap),
      gamma_(gamma),
      gamma_inv_(1.0 / gamma)
{
    auto suffix = GetSuffix(filename);
    if (!kStbInputFormat.count(suffix))
    {
        std::cerr << "[error] unsupport input image format \"" << suffix
                  << "\" for image:" << filename << std::endl;
        exit(1);
    }

    switch (Hash(suffix.c_str()))
    {
    case "exr"_hash:
    case "EXR"_hash:
        LoadByTinyExr(filename, gamma, data_, width_, height_, channels_);
        break;
    default:
        LoadByStbImage(filename, gamma, data_, width_, height_, channels_);
        break;
    }
}

///\return 纹理在给定坐标处梯度
Spectrum Bitmap::Color(const Vector2 &coord) const
{
    auto x = static_cast<int>(coord.x * width_),
         y = static_cast<int>(coord.y * height_);
    x = Modulo(x, width_);
    y = Modulo(y, height_);
    auto offset = (x + static_cast<size_t>(width_) * y) * channels_;
    if (channels_ == 1)
        return Spectrum(data_[offset], data_[offset], data_[offset]);
    else
    {
        auto r = data_[offset];
        auto g = data_[offset + 1];
        auto b = data_[offset + 2];
        return Spectrum(r, g, b);
    }
}

///\return 纹理在给定坐标处梯度
Vector2 Bitmap::Gradient(const Vector2 &coord) const
{
    auto GetNorm = [&](const Vector2 &coord, int offset_x, int offset_y)
    {
        auto x = static_cast<int>(coord.x * width_) + offset_x,
             y = static_cast<int>(coord.y * height_) + offset_y;
        x = Modulo(x, width_);
        y = Modulo(y, height_);
        auto offset = (x + static_cast<size_t>(width_) * y) * channels_;
        if (channels_ == 1)
            return 255 * data_[offset];
        else
        {
            auto r = 255 * data_[offset];
            auto g = 255 * data_[offset + 1];
            auto b = 255 * data_[offset + 2];
            return glm::length(Vector3(r, g, b));
        }
    };
    Float kh = 0.2, kn = 0.1;
    auto value = GetNorm(coord, 0, 0),
         value_u = GetNorm(coord, 1, 0),
         value_v = GetNorm(coord, 0, 1);
    auto du = kh * kn * (value_u - value),
         dv = kh * kn * (value_v - value);
    return Vector2(du, dv);
}

///\return 材质在给定的纹理坐标处是否透明
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

///\brief 设置纹理在给定坐标处像素值
void Bitmap::SetColor(int x, int y, const Vector3 &value)
{
    auto offset = (static_cast<size_t>(x) + width_ * y) * channels_;
    for (int i = 0; i < 3; i++)
        data_[offset + i] = ClampBottom<Float>(0, value[i]);
}

///\brief 保存图像到指定路径
void Bitmap::Save(const std::string &filename)
{
    auto suffix = GetSuffix(filename);
    switch (Hash(suffix.c_str()))
    {
    case "exr"_hash:
    case "EXR"_hash:
        SaveHdrByTinyExr(filename, data_, width_, height_);
        break;
    case "hdr"_hash:
    case "HDR"_hash:
    {
        auto raw_data = std::vector<float>(data_.begin(), data_.end());
        stbi_write_hdr(filename.c_str(), width_, height_, channels_, raw_data.data());
        break;
    }
    default:
        SaveSdrByStbImage(filename, gamma_inv_, data_, width_, height_);
        break;
    }
}

///\brief 撤销伽马校正
Float UndoGamma(Float value, Float gamma)
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

///\brief 应用伽马校正
Float ApplyGamma(Float value, Float gamma_inv)
{
    if (gamma_inv == -1)
        return (value <= (Float)0.0031308) ? ((Float)12.92 * value)
                                           : ((Float)1.055 * std::pow(value, (Float)(1.0 / 2.4)) - (Float)0.055);
    else
        return std::pow(value, gamma_inv);
}

///\brief 通过 tinyexr 加载 HDR 图像
void LoadByTinyExr(const std::string &filename, Float gamma, std::vector<Float> &data, int &width, int &height, int &channels)
{
    float *raw_data; // width * height * RGBA
    const char *err = nullptr;
    int ret = LoadEXR(&raw_data, &width, &height, filename.c_str(), &err);
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
    channels = 4;
    auto cnt = width * height * channels;
    data.resize(cnt);
    for (size_t i = 0; i < cnt; i++)
        data[i] = UndoGamma(raw_data[i], gamma);
    free(raw_data);
    raw_data = nullptr;
}

///\brief 通过 stb 加载图像
void LoadByStbImage(const std::string &filename, Float gamma, std::vector<Float> &data, int &width, int &height, int &channels)
{
    auto raw_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (raw_data == nullptr)
    {
        std::cerr << "[error] load image \"" << filename << "\" failed." << std::endl;
        exit(1);
    }
    auto cnt = width * height * channels;
    data.resize(cnt);
    for (size_t i = 0; i < cnt; i++)
        data[i] = UndoGamma(static_cast<float>(raw_data[i]) / 255.0, gamma);
    stbi_image_free(raw_data);
    raw_data = nullptr;
}

///\brief 通过 stb 保存 SDR 图像
void SaveSdrByStbImage(const std::string &filename, Float gamma_inv, const std::vector<Float> &data, int width, int height)
{
    auto raw_data = std::vector<unsigned char>(width * height * 3);
    for (size_t i = 0; i < data.size(); i++)
        raw_data[i] = static_cast<unsigned char>(ClampTop<int>(255, 255 * ApplyGamma(data[i], gamma_inv)));
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
        std::cout << "[warning] unsupported output format \"" << suffix << "\", use png instead." << std::endl;
        stbi_write_png(ChangeSuffix(filename, "png").c_str(), width, height, 3, raw_data.data(), width * 3);
        break;
    }
}

///\brief 通过 tinyexr 保存 HDR 图像
void SaveHdrByTinyExr(const std::string &filename, const std::vector<Float> &data, int width, int height)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    auto resolution = width * height;

    images[0].resize(resolution);
    images[1].resize(resolution);
    images[2].resize(resolution);

    for (size_t i = 0; i < resolution; i++)
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

NAMESPACE_END(simple_renderer)