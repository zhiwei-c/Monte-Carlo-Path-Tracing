#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的理想漫反射材质派生类
class Diffuse : public Material
{
public:
    ///\brief 平滑的理想漫反射材质
    ///\param id 材质id
    ///\param reflectance 漫反射系数
    Diffuse(std::unique_ptr<Texture> reflectance)
        : Material(MaterialType::kDiffuse),
          reflectance_(std::move(reflectance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override
    {
        SampleHemisCos(bs.normal, bs.wi, &bs.pdf);
        if (!bs.get_attenuation)
            return;
        bs.attenuation = reflectance_->Color(bs.texcoord) * kPiInv;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
                  bool inside) const override
    {
        return reflectance_->Color(texcoord) * kPiInv;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord,
              bool inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        else
            return PdfHemisCos(wo, normal);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               !reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Material::Transparent(texcoord) ||
               reflectance_->Transparent(texcoord);
    }

private:
    std::unique_ptr<Texture> reflectance_; //漫反射系数
};

NAMESPACE_END(raytracer)