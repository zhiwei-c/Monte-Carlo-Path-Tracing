#pragma once

#include "../global.hpp"
#include "../utils/image.hpp"

NAMESPACE_BEGIN(raytracer)

//纹理的类型
enum class TextureType
{
    kConstant,     //恒定的颜色
    kBitmap,       //位图
    kCheckerboard, //棋盘纹理
    kGridTexture,  //网格纹理
};

//纹理
class Texture
{
public:
    virtual ~Texture() {}

    virtual int width() const { return -1; }
    virtual int height() const { return -1; }
    const std::string &id() const { return id_; }

    double luminance(const dvec2 &texcoord) const { return LinearRgbToLuminance(color(texcoord)); }
    virtual dvec3 color(const dvec2 &texcoord) const = 0;
    virtual dvec2 gradient(const dvec2 &texcoord) const = 0;
    bool IsConstant() const { return type_ == TextureType::kConstant; }
    bool IsBitmap() const { return type_ == TextureType::kBitmap; }
    virtual bool IsTransparent(const dvec2 &texcoord, Sampler* sampler) const { return false; }

protected:
    Texture(TextureType type, const std::string &id) : type_(type), id_(id) {}

private:
    TextureType type_; //纹理类型
    std::string id_;   //纹理ID
};

NAMESPACE_END(raytracer)