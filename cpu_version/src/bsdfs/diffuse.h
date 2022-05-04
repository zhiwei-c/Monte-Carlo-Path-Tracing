#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的理想漫反射材质派生类
class Diffuse : public Material
{
public:
    ///\brief 平滑的理想漫反射材质
    ///\param id 材质id
    ///\param reflectance 漫反射系数
    Diffuse(std::unique_ptr<Texture> reflectance);

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
    std::unique_ptr<Texture> reflectance_; //漫反射系数
};

NAMESPACE_END(simple_renderer)