#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 冯模型描述的有光泽材质派生类
class Glossy : public Material
{
public:
    ///\brief 冯模型描述的有光泽材质
    ///\param diffuse_reflectance 漫反射系数
    ///\param specular_reflectance 镜面反射系数
    ///\param exponent 镜面反射指数系数
    Glossy(std::unique_ptr<Texture> diffuse_reflectance,
           std::unique_ptr<Texture> specular_reflectance,
           Float exponent);

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override;

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 是否映射纹理
    bool TextureMapping() const override;

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override;

private:
    Float exponent_;                                //镜面反射指数系数
    Float diffuse_reflectance_sum_;                 //漫反射系数和
    Float specular_reflectance_sum_;                //镜面反射系数和
    Float diffuse_sampling_weight_;                 //抽样漫反射权重
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数

    ///\brief 获取给定点抽样漫反射的权重
    Float DiffuseSamplingWeight(const Vector2 &texcoord) const;
};

NAMESPACE_END(simple_renderer)