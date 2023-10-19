#pragma once

#include "../tensor/tensor.cuh"

class Texture
{
public:
    enum Type
    {
        kConstant,
        kCheckerboard,
        kBitmap,
    };

    struct Info
    {
        Type type;
        struct Data
        {
            struct Constant
            {
                Vec3 color;
            } constant;

            struct Checkerboard
            {
                Vec3 color0;
                Vec3 color1;
                Mat4 to_uv;
            } checkerboard;

            struct Bitmap
            {
                uint32_t offset;
                int width;
                int height;
                int channel;
            } bitmap;

        } data;

        static Info CreateConstant(const Vec3 &color);
        static Info CreateCheckerboard(const Vec3 &color0, const Vec3 &color1, const Mat4 &to_uv);
        static Info CreateBitmap(const uint32_t offset, const int width, const int height,
                                 const int channel);
    };

    QUALIFIER_DEVICE virtual ~Texture() {}

    QUALIFIER_DEVICE virtual Vec3 GetColor(const Vec2 &texcoord,
                                           const float *pixel_buffer) const = 0;
    QUALIFIER_DEVICE virtual Vec2 GetGradient(const Vec2 &texcoord,
                                              const float *pixel_buffer) const = 0;

    QUALIFIER_DEVICE virtual bool IsTransparent(const Vec2 &texcoord, const float *pixel_buffer,
                                                uint32_t *seed) const
    {
        return false;
    }

protected:
    QUALIFIER_DEVICE Texture(uint32_t id, Type type) : id_(id), type_(type) {}

private:
    uint32_t id_;
    Type type_;
};

class ConstantTexture : public Texture
{
public:
    QUALIFIER_DEVICE ConstantTexture(uint32_t id, const Info::Data::Constant &data)
        : Texture(id, Type::kConstant), color_(data.color)
    {
    }

    QUALIFIER_DEVICE Vec3 GetColor(const Vec2 &texcoord, const float *pixel_buffer) const override
    {
        return color_;
    }

    QUALIFIER_DEVICE Vec2 GetGradient(const Vec2 &texcoord,
                                      const float *pixel_buffer) const override
    {
        return {0, 0};
    }

private:
    Vec3 color_;
};

class CheckerboardTexture : public Texture
{
public:
    QUALIFIER_DEVICE CheckerboardTexture(uint32_t id, const Info::Data::Checkerboard &data)
        : Texture(id, Type::kCheckerboard), color0_(data.color0), color1_(data.color1),
          to_uv_(data.to_uv)
    {
    }

    QUALIFIER_DEVICE Vec3 GetColor(const Vec2 &texcoord, const float *pixel_buffer) const override;
    QUALIFIER_DEVICE Vec2 GetGradient(const Vec2 &texcoord,
                                      const float *pixel_buffer) const override;

private:
    Vec3 color0_;
    Vec3 color1_;
    Mat4 to_uv_;
};

class Bitmap : public Texture
{
public:
    QUALIFIER_DEVICE Bitmap(uint32_t id, const Info::Data::Bitmap &data)
        : Texture(id, Type::kBitmap), offset_(data.offset), width_(data.width),
          height_(data.height), channel_(data.channel)
    {
    }

    QUALIFIER_DEVICE Vec3 GetColor(const Vec2 &texcoord, const float *pixel_buffer) const override;
    QUALIFIER_DEVICE Vec2 GetGradient(const Vec2 &texcoord,
                                      const float *pixel_buffer) const override;

    QUALIFIER_DEVICE bool IsTransparent(const Vec2 &texcoord, const float *pixel_buffer,
                                        uint32_t *seed) const override;

private:
    uint32_t offset_;
    int width_;
    int height_;
    int channel_;
};
