#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的导体材质派生类
class Conductor : public Bsdf
{
public:
    ///\brief 平滑的导体材质
    ///\param eta 材质折射率的实部
    ///\param k 材质折射率的虚部（消光系数）
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Conductor(const Spectrum &eta, const Spectrum &k, Float ext_ior, std::unique_ptr<Texture> specular_reflectance)
        : Bsdf(BsdfType::kConductor), eta_(eta / ext_ior), k_(k / ext_ior),
          specular_reflectance_(std::move(specular_reflectance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        //生成光线方向，计算光线传播概率
        rec.wi = -Reflect(-rec.wo, rec.normal);
        rec.pdf = 1;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        if (!rec.get_attenuation)
            return;
        rec.attenuation = FresnelConductor(rec.wi, rec.normal, eta_, k_) * glm::dot(-rec.wi, rec.normal);
        if (specular_reflectance_)
            rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    void Eval(SamplingRecord &rec) const override
    {
        if (!SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
        {
            return;
        }
        rec.pdf = 1;
        rec.type = ScatteringType::kReflect;
        rec.attenuation = FresnelConductor(rec.wi, rec.normal, eta_, k_) * glm::dot(-rec.wi, rec.normal);
        if (specular_reflectance_)
            rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && !specular_reflectance_->Constant();
    }

private:
    Spectrum eta_;                                  //材质相对折射率的实部
    Spectrum k_;                                    //材质相对折射率的虚部（消光系数）
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)