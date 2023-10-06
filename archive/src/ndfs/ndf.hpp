#pragma once

#include <array>
#include <utility>

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

// 法线分布函数类型
enum class NdfType
{
    kBeckmann,
    kGgx,
};

constexpr int kLutResolution = 512; //预计算纹理贴图的精度

//法线分布函数
class Ndf
{
public:
    virtual ~Ndf() {}

    void ComputeAlbedoTable();

    virtual void Sample(const dvec3 &n, double alpha_u, double alpha_v, const dvec2 &sample, dvec3 *h, double *pdf) const = 0;
    virtual double Pdf(const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const = 0;
    virtual double SmithG1(const dvec3 &v, const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const = 0;

    double albedo_avg() const { return albedo_avg_; }
    bool UseCompensation() const { return compensate_; }

    bool UseTextureMapping() const;
    double albdo(double cos_theta) const;
    std::pair<double, double> roughness(const dvec2 &texcoord) const;

protected:
    Ndf(NdfType type, Texture *alpha_u, Texture *alpha_v);

    Texture *alpha_u_; //粗糙度
    Texture *alpha_v_; //粗糙度

private:
    NdfType type_;                                  //材质类型（表面散射模型类型）
    bool compensate_;                               //是否补偿在微表面之间多次散射后又射出的能量
    double albedo_avg_;                             //平均反照率
    std::array<double, kLutResolution> albedo_lut_; //反照率查找表
};

// GGX 法线分布函数
class GgxNdf : public Ndf
{
public:
    GgxNdf(Texture *alpha_u, Texture *alpha_v) : Ndf(NdfType::kGgx, alpha_u, alpha_v) {}

    void Sample(const dvec3 &n, double alpha_u, double alpha_v, const dvec2 &sample, dvec3 *h, double *pdf) const override;
    double Pdf(const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const override;
    double SmithG1(const dvec3 &v, const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const override;
};

// Beckmann 法线分布函数
class BeckmannNdf : public Ndf
{
public:
    BeckmannNdf(Texture *alpha_u, Texture *alpha_v) : Ndf(NdfType::kBeckmann, alpha_u, alpha_v) {}

    void Sample(const dvec3 &n, double alpha_u, double alpha_v, const dvec2 &sample, dvec3 *h, double *pdf) const override;
    double Pdf(const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const override;
    double SmithG1(const dvec3 &v, const dvec3 &h, const dvec3 &n, double alpha_u, double alpha_v) const override;
};

NAMESPACE_END(raytracer)