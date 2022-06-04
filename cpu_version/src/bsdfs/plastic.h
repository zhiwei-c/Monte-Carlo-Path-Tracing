#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 平滑的塑料材质派生类
class Plastic : public Bsdf
{
public:
    ///\brief 平滑的塑料材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param diffuse_reflectance 漫反射系数
    ///\param nonlinear 是否考虑因内部散射而引起的非线性色移
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Plastic(Float int_ior, Float ext_ior, std::unique_ptr<Texture> diffuse_reflectance,
            std::unique_ptr<Texture> specular_reflectance, bool nonlinear)
        : Bsdf(BsdfType::kPlastic), eta_inv_(ext_ior / int_ior), fdr_(AverageFresnel(int_ior / ext_ior)),
          diffuse_reflectance_(std::move(diffuse_reflectance)), specular_reflectance_(std::move(specular_reflectance)),
          nonlinear_(nonlinear), specular_sampling_weight_(-1)
    {
        if (!diffuse_reflectance_->Constant() || specular_reflectance_ && !specular_reflectance_->Constant())
            return;
        Spectrum kd = diffuse_reflectance_->Color(Vector2(0));
        Float s_sum = 3, d_sum = kd.r + kd.g + kd.b;
        if (specular_reflectance_)
        {
            Spectrum ks = specular_reflectance_->Color(Vector2(0));
            s_sum = ks.r + ks.g + ks.b;
        }
        specular_sampling_weight_ = s_sum / (d_sum + s_sum);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        bool sampled_specular = false;                                       //是否抽样到了镜面反射分量
        Float kr_i = 0,                                                      //入射菲涅尔项
            kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_),                   //出射菲涅尔项
            specular_sampling_weight = SpecularSamplingWeight(rec.texcoord), //抽样镜面反射的权重
            pdf_specular = kr_o * specular_sampling_weight,                  //抽样镜面反射分量的概率
            pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);   //抽样漫反射分量的概率
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = 0;
        if (UniformFloat() < pdf_specular)
        { //从镜面反射分量抽样光线方向
            rec.wi = -Reflect(-rec.wo, rec.normal);
            kr_i = kr_o;
            rec.pdf += pdf_specular;
            sampled_specular = true;
        }
        else
        { //从漫反射分量抽样光线方向
            SampleHemisCos(rec.normal, rec.wi);
            kr_i = Fresnel(rec.wi, rec.normal, eta_inv_);
            pdf_specular = kr_i * specular_sampling_weight;
            pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
            pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        }
        rec.pdf += (1.0 - pdf_specular) * PdfHemisCos(rec.wo, rec.normal);
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        if (!rec.get_attenuation)
            return;
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(rec.texcoord);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_);
        rec.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (sampled_specular)
            rec.attenuation += kr_i * (specular_reflectance_ ? specular_reflectance_->Color(rec.texcoord) : Spectrum(1));
        //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
        rec.attenuation *= glm::dot(-rec.wi, rec.normal);
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
        //计算光线传播概率
        bool sampled_specular = false;                                       //是否抽样到了镜面反射分量
        Float kr_i = Fresnel(rec.wi, rec.normal, eta_inv_),                  //入射菲涅尔项
            specular_sampling_weight = SpecularSamplingWeight(rec.texcoord), //抽样镜面反射的权重
            pdf_specular = kr_i * specular_sampling_weight,                  //抽样镜面反射分量的概率
            pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);   //抽样漫反射分量的概率
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(rec.wo, rec.normal);
        if (SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
        { //如果出射方向位于镜面反射波瓣之内，则再加上镜面反射成分的概率
            rec.pdf += pdf_specular;
            sampled_specular = true;
        }
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        //计算光能衰减系数
        Spectrum diffuse_reflectance = diffuse_reflectance_->Color(rec.texcoord);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (static_cast<Float>(1) - diffuse_reflectance * fdr_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_);
        Float kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_); //出射菲涅尔项
        rec.attenuation *= Sqr(eta_inv_) * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (sampled_specular)
            rec.attenuation += kr_i * (specular_reflectance_ ? specular_reflectance_->Color(rec.texcoord) : Spectrum(1));
        //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
        rec.attenuation *= glm::dot(-rec.wi, rec.normal);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || !diffuse_reflectance_->Constant() || specular_reflectance_ && !specular_reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Bsdf::Transparent(texcoord) || diffuse_reflectance_->Transparent(texcoord);
    }

private:
    ///\brief 获取给定点抽样镜面反射的权重
    Float SpecularSamplingWeight(const Vector2 &texcoord) const
    {
        if (specular_sampling_weight_ >= 0)
            return specular_sampling_weight_;
        Spectrum kd = diffuse_reflectance_->Color(texcoord);
        Float d_sum = kd.r + kd.g + kd.b;
        if (!specular_reflectance_)
            return 3.0 / (d_sum + 3.0);
        Spectrum ks = specular_reflectance_->Color(texcoord);
        Float s_sum = ks.r + ks.g + ks.b;
        return s_sum / (d_sum + s_sum);
    }

    bool nonlinear_;                                //是否考虑因内部散射而引起的非线性色移
    Float specular_sampling_weight_;                //抽样镜面反射的权重
    Float fdr_;                                     //漫反射菲涅尔项的平均值
    Float eta_inv_;                                 //外部折射率与介质折射率之比
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(raytracer)