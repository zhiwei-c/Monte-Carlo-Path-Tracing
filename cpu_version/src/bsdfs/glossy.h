#pragma once

#include "../core/bsdf_base.h"

NAMESPACE_BEGIN(raytracer)

///\brief 冯模型描述的有光泽材质派生类
class Glossy : public Bsdf
{
public:
    ///\brief 冯模型描述的有光泽材质
    ///\param diffuse_reflectance 漫反射系数
    ///\param specular_reflectance 镜面反射系数
    ///\param exponent 镜面反射指数系数
    Glossy(std::unique_ptr<Texture> diffuse_reflectance, std::unique_ptr<Texture> specular_reflectance, Float exponent)
        : Bsdf(BsdfType::kGlossy), diffuse_reflectance_(std::move(diffuse_reflectance)),
          specular_reflectance_(std::move(specular_reflectance)), exponent_(exponent), diffuse_sampling_weight_(-1),
          diffuse_reflectance_sum_(-1), specular_reflectance_sum_(-1)
    {
        if (diffuse_reflectance_->Constant())
        {
            Spectrum kd = diffuse_reflectance->Color(Vector2(0));
            diffuse_reflectance_sum_ = kd.r + kd.g + kd.b;
        }
        if (specular_reflectance->Constant())
        {
            Spectrum ks = specular_reflectance->Color(Vector2(0));
            specular_reflectance_sum_ = ks.r + ks.g + ks.b;
        }
        if (diffuse_reflectance_->Constant() && specular_reflectance->Constant())
            diffuse_sampling_weight_ = diffuse_reflectance_sum_ / (diffuse_reflectance_sum_ + specular_reflectance_sum_);
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(SamplingRecord &rec) const override
    {
        Float pdf_diffuse = DiffuseSamplingWeight(rec.texcoord); //抽样漫反射的权重
        if (UniformFloat() < pdf_diffuse)
        { //抽样漫反射分量
            SampleHemisCos(rec.normal, rec.wi);
        }
        else
        { //抽样镜面反射反射分量
            rec.wi = SampleHemisCosN(exponent_, rec.normal);
            if (SameHemis(rec.wi, rec.normal))
                return;
        }
        Eval(rec);
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
        Float pdf_diffuse = DiffuseSamplingWeight(rec.texcoord), //抽样漫反射的权重
            pdf = pdf_diffuse * PdfHemisCos(rec.wo, rec.normal); //漫反射成分的概率
        Vector3 wr = Reflect(rec.wi, rec.normal);                //理想镜面反射方向
        Float cos_alpha = glm::dot(rec.wo, wr);                  //出射方向和理想镜面反射方向夹角的余弦
        if (cos_alpha > 0)
        { //如果出射方向位于镜面反射波瓣之内，则再加上镜面反射成分的概率
            pdf += (1.0 - pdf_diffuse) * PdfHemisCosN(rec.wo, wr, exponent_);
        }
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.type = ScatteringType::kReflect;
        
        if (!rec.get_attenuation)
            return;
        auto albedo = Spectrum(0);
        // 计算漫反射分量的贡献
        rec.attenuation = diffuse_reflectance_->Color(rec.texcoord) * kPiInv;
        // 计算镜面反射分量的贡献
        if (cos_alpha > 0)
            rec.attenuation += specular_reflectance_->Color(rec.texcoord) *
                               static_cast<Float>((exponent_ + 2) * kPiInv * 0.5 * std::pow(cos_alpha, exponent_));
        //因为 BSDF 是入射辐射照度和出射辐射亮度之间的关系，所以需要乘以入射方向和表面方向夹角的余弦，将入射辐射亮度转换为入射辐射照度.
        rec.attenuation *= glm::dot(-rec.wi, rec.normal);
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || !diffuse_reflectance_->Constant() || !specular_reflectance_->Constant();
    }

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override
    {
        return Bsdf::Transparent(texcoord) || diffuse_reflectance_->Transparent(texcoord);
    }

private:
    ///\brief 获取给定点抽样漫反射的权重
    Float DiffuseSamplingWeight(const Vector2 &texcoord) const
    {
        if (diffuse_sampling_weight_ >= 0)
            return diffuse_sampling_weight_;

        Float ks_sum = specular_reflectance_sum_;
        if (ks_sum < 0)
        {
            Spectrum ks = specular_reflectance_->Color(texcoord);
            ks_sum = ks.r + ks.g + ks.b;
        }

        Float kd_sum = diffuse_reflectance_sum_;
        if (kd_sum < 0)
        {
            Spectrum kd = diffuse_reflectance_->Color(texcoord);
            kd_sum = kd.r + kd.g + kd.b;
        }

        return kd_sum / (kd_sum + ks_sum);
    }

    Float exponent_;                                //镜面反射指数系数
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数
    Float diffuse_reflectance_sum_;                 //漫反射系数和
    Float specular_reflectance_sum_;                //镜面反射系数和
    Float diffuse_sampling_weight_;                 //抽样漫反射权重
};

NAMESPACE_END(raytracer)