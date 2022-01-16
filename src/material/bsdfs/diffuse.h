#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

class Diffuse : public Material
{
public:
    /**
     * \brief 漫反射材质
     * \param id 材质id
     * \param reflectance 漫反射系数
     * \param ks 镜面反射系数
     * \param ns 镜面反射指数系数
     * \param diffuse_map_name 用于漫反射纹理的图片路径
     */
    Diffuse(const std::string &id, const Spectrum &reflectance, Texture *diffuse_map)
        : Material(id, MaterialType::kDiffuse), reflectance_(reflectance), diffuse_map_(diffuse_map) {}

    ~Diffuse()
    {
        if (diffuse_map_)
            DeleteTexturePointer(diffuse_map_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        BsdfSampling bs;

        auto [wi_local, pdf] = HemisCos();
        if (pdf < kEpsilon)
            return BsdfSampling();

        bs.wi = -ToWorld(wi_local, normal);
        bs.pdf = pdf;

        if (texcoord != nullptr && diffuse_map_)
            bs.weight = diffuse_map_->GetPixel(*texcoord) * kPiInv;
        else
            bs.weight = reflectance_ * kPiInv;

        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return Spectrum(0);

        if (texcoord != nullptr && diffuse_map_)
            return diffuse_map_->GetPixel(*texcoord) * kPiInv;
        else
            return reflectance_ * kPiInv;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto wo_local = ToLocal(wo, normal);
        return PdfHemisCos(wo_local);
    }

    bool TextureMapping() const override { return diffuse_map_ != nullptr; }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else if (diffuse_map_)
            return diffuse_map_->Transparent(texcoord);
        else
            return false;
    }

private:
    Spectrum reflectance_;  //漫反射系数
    Texture *diffuse_map_; //纹理，用于映射漫反射系数
};

NAMESPACE_END(simple_renderer)