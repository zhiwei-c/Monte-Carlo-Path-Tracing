#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的电介质派生类
class Dielectric : public Bsdf
{
public:
    ///\brief 平滑的电介质材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Dielectric(Float int_ior, Float ext_ior, std::unique_ptr<Texture> specular_reflectance,
               std::unique_ptr<Texture> specular_transmittance)
        : Bsdf(BsdfType::kDielectric), eta_(int_ior / ext_ior), eta_inv_(ext_ior / int_ior),
          specular_reflectance_(std::move(specular_reflectance)), specular_transmittance_(std::move(specular_transmittance))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        Float eta = rec.inside ? eta_inv_ : eta_,       //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
            eta_inv = rec.inside ? eta_ : eta_inv_,     //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(-rec.wo, rec.normal, eta_inv); //菲涅尔项
        if (UniformFloat() < kr)
        { //抽样反射光线
            //计算光线传播概率
            rec.pdf = kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.type = ScatteringType::kReflect;
            //生成光线方向
            rec.wi = -Reflect(-rec.wo, rec.normal);
            //计算光能衰减系数
            rec.attenuation = Spectrum(kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else
        { //抽样折射光线
            //生成光线方向
            rec.wi = -Refract(-rec.wo, rec.normal, eta_inv);
            { //光线折射时穿过了介质，为了使实际的入射光线方向和表面法线方向夹角的余弦仍小于零，需做一些相应处理
                rec.normal = -rec.normal;
                rec.inside = !rec.inside;
                eta_inv = eta;
            }
            kr = Fresnel(rec.wi, rec.normal, eta_inv);
            //计算光线传播概率
            rec.pdf = 1.0 - kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.type = ScatteringType::kTransimission;
            //计算光能衰减系数
            rec.attenuation = Spectrum(1.0 - kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= Sqr(eta_inv);
        }
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    void Eval(SamplingRecord &rec) const override
    {
        Float eta_inv = rec.inside ? eta_ : eta_inv_,  //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(rec.wi, rec.normal, eta_inv); //菲涅尔项
        if (SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
        { //处理镜面反射
            rec.pdf = kr;
            rec.type = ScatteringType::kReflect;
            rec.attenuation = Spectrum(kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else if (SameDirection(rec.wo, Refract(rec.wi, rec.normal, eta_inv)))
        { //处理折射
            rec.pdf = 1.0 - kr;
            rec.type = ScatteringType::kTransimission;
            rec.attenuation = Spectrum(1.0 - kr) * glm::dot(-rec.wi, rec.normal);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= Sqr(eta_inv);
        }
        else
            return;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && !specular_reflectance_->Constant() ||
               specular_transmittance_ && !specular_transmittance_->Constant();
    }

private:
    Float eta_;                                       //介质折射率与外部折射率之比
    Float eta_inv_;                                   //外部折射率与介质折射率之比
    std::unique_ptr<Texture> specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    std::unique_ptr<Texture> specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)