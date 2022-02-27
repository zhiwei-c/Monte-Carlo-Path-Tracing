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
    Diffuse(const std::string &id, Texture *reflectance)
        : Material(id, MaterialType::kDiffuse), reflectance_(reflectance) {}

    ~Diffuse()
    {
        delete reflectance_;
        reflectance_ = nullptr;
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
    {
        BsdfSampling bs;

        auto [wi_local, pdf] = HemisCos();
        if (pdf < kEpsilonL)
            return BsdfSampling();

        bs.wi = -ToWorld(wi_local, normal);
        bs.pdf = pdf;

        if (get_weight)
        {
            if (texcoord != nullptr)
                bs.weight = reflectance_->GetPixel(*texcoord) * kPiInv;
            else
                bs.weight = reflectance_->GetPixel(Vector2(0)) * kPiInv;
        }

        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return Spectrum(0);

        if (texcoord != nullptr)
            return reflectance_->GetPixel(*texcoord) * kPiInv;
        else
            return reflectance_->GetPixel(Vector2(0)) * kPiInv;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (NotSameHemis(wo, normal))
            return 0;

        auto wo_local = ToLocal(wo, normal);
        return PdfHemisCos(wo_local);
    }

    bool TextureMapping() const override { return !reflectance_->Constant(); }

    bool Transparent(const Vector2 &texcoord) const override
    {
        if (Material::Transparent(texcoord))
            return true;
        else
            return reflectance_->Transparent(texcoord);
    }

private:
    Texture *reflectance_; //漫反射系数
};

NAMESPACE_END(simple_renderer)