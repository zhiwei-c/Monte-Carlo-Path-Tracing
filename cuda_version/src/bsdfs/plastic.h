#pragma once

#include "../core/bsdf.h"

///\brief 平滑的塑料材质派生类
class Plastic : public Bsdf
{
public:
    /**
     * @brief 平滑的塑料材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param eta 相对折射率
     * @param diffuse_reflectance 漫反射系数
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param nonlinear 是否考虑因内部散射而引起的非线性色移
     */
    __device__ Plastic(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map, vec3 eta, Texture *diffuse_reflectance,
                       Texture *specular_reflectance, bool nonlinear)
        : Bsdf(idx, kPlastic, twosided, bump_map, opacity_map), eta_inv_d_(1.0 / eta.x), diffuse_reflectance_(diffuse_reflectance),
          specular_reflectance_(specular_reflectance), nonlinear_(nonlinear), fdr_int_(AverageFresnel(eta.x))
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const override
    {
        Float kr_i = 0, kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_d_),
              specular_sampling_weight = SpecularSamplingWeight(rec.texcoord);
        rec.pdf = 0;
        Float pdf_specular = kr_o * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        auto specular = false;
        if (sample.x < pdf_specular)
        {
            rec.wi = -Reflect(-rec.wo, rec.normal);
            kr_i = kr_o;
            rec.pdf += pdf_specular;
            specular = true;
        }
        else
        {
            auto wi_local = vec3(0);
            Float pdf = 0;
            HemisCos(sample.y, sample.z, wi_local, pdf);
            rec.wi = -ToWorld(wi_local, rec.normal);
            kr_i = Fresnel(rec.wi, rec.normal, eta_inv_d_);
            pdf_specular = kr_i * specular_sampling_weight;
            pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
            pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        }
        pdf_diffuse = 1.0 - pdf_specular;
        rec.pdf += pdf_diffuse * PdfHemisCos(ToLocal(rec.wo, rec.normal));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        vec3 diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(rec.texcoord) : vec3(0.5);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_int_);
        rec.attenuation *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (specular)
        {
            auto attenuation = vec3(kr_i);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += attenuation;
        }
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(rec.wo, rec.normal))
            return;
        Float kr_i = Fresnel(rec.wi, rec.normal, eta_inv_d_),
              specular_sampling_weight = SpecularSamplingWeight(rec.texcoord);
        Float pdf_specular = kr_i * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(ToLocal(rec.wo, rec.normal));
        if (SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
            rec.pdf += pdf_specular;
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        vec3 diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(rec.texcoord) : vec3(0.5);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_int_);
        Float kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_d_);
        rec.attenuation *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        if (SameDirection(Reflect(rec.wi, rec.normal), rec.wo))
        {
            auto attenuation = vec3(kr_i);
            if (specular_reflectance_)
                attenuation *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += attenuation;
        }
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || diffuse_reflectance_ && diffuse_reflectance_->Varying() ||
               specular_reflectance_ && specular_reflectance_->Varying();
    }

    ///\return 给定点是否透明
    __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const override
    {
        if (Bsdf::Transparent(texcoord, sample))
            return true;
        else if (!diffuse_reflectance_ || !diffuse_reflectance_->IsBitmap() ||
                 !diffuse_reflectance_->Transparent(texcoord, sample.y))
            return false;
        else
            return true;
    }

private:
    ///\return 给定点抽样镜面反射的权重
    __device__ Float SpecularSamplingWeight(const vec2 &texcoord) const
    {
        if (!diffuse_reflectance_ && !specular_reflectance_)
            return 2.0 / 3.0;
        else if (!diffuse_reflectance_)
        {
            vec3 ks = specular_reflectance_->Color(texcoord);
            Float s_sum = ks.x + ks.y + ks.z;
            return s_sum / (1.5 + s_sum);
        }
        else if (!specular_reflectance_)
        {
            vec3 kd = diffuse_reflectance_->Color(texcoord);
            return 3.0 / (kd.x + kd.y + kd.z + 3.0);
        }
        else
        {
            vec3 ks = specular_reflectance_->Color(texcoord);
            Float s_sum = ks.x + ks.y + ks.z;
            vec3 kd = diffuse_reflectance_->Color(texcoord);
            return s_sum / (kd.x + kd.y + kd.z + s_sum);
        }
    }

    Float eta_inv_d_;               //外部折射率与介质折射率之比
    Texture *diffuse_reflectance_;  //漫反射系数
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    bool nonlinear_;                //是否考虑因内部散射而引起的非线性色移
    Float fdr_int_;                 //漫反射菲涅尔项平均值
};

__device__ inline void InitPlastic(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
                                   Bsdf **&bsdf_list)
{
    Texture *bump_map = nullptr;
    if (bsdf_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + bsdf_info_list[m_idx].bump_map_idx;

    Texture *opacity_map = nullptr;
    if (bsdf_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + bsdf_info_list[m_idx].opacity_idx;

    Texture *diffuse_reflectance = nullptr;
    if (bsdf_info_list[m_idx].diffuse_reflectance_idx != kUintMax)
        diffuse_reflectance = texture_list + bsdf_info_list[m_idx].diffuse_reflectance_idx;

    Texture *specular_reflectance = nullptr;
    if (bsdf_info_list[m_idx].specular_reflectance_idx != kUintMax)
        specular_reflectance = texture_list + bsdf_info_list[m_idx].specular_reflectance_idx;

    bsdf_list[m_idx] = new Plastic(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                   bsdf_info_list[m_idx].eta, diffuse_reflectance, specular_reflectance,
                                   bsdf_info_list[m_idx].nonlinear);
}