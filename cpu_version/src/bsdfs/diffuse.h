#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的理想漫反射材质派生类
class Diffuse : public Bsdf
{
public:
    ///\brief 平滑的理想漫反射材质
    ///\param id 材质id
    ///\param reflectance 漫反射系数
    Diffuse(std::unique_ptr<Texture> reflectance)
        : Bsdf(BsdfType::kDiffuse), reflectance_(std::move(reflectance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        //生成光线方向，计算光线传播概率
        SampleHemisCos(rec.normal, rec.wi, &rec.pdf);
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        if (!rec.get_attenuation)
            return;
        rec.attenuation = reflectance_->Color(rec.texcoord) * kPiInv * glm::dot(-rec.wi, rec.normal);
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    void Eval(SamplingRecord &rec) const override
    {
        if (glm::dot(rec.wo, rec.normal) < 0)
        { //表面法线方向、光线入射和出射方向需在介质同侧，否则没有贡献，
            //又因为数据传入时已处理光线入射方向和表面法线方向，使两者在介质同侧，
            //故只需确保光线出射方向和表面法线方向在介质同侧即可
            return;
        }
        rec.pdf = PdfHemisCos(rec.wo, rec.normal);
        rec.type = ScatteringType::kReflect;
        rec.attenuation = reflectance_->Color(rec.texcoord) * kPiInv * glm::dot(-rec.wi, rec.normal);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || !reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Bsdf::Transparent(texcoord) || reflectance_->Transparent(texcoord);
    }

private:
    std::unique_ptr<Texture> reflectance_; //漫反射系数
};

NAMESPACE_END(raytracer)