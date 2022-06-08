#pragma once

#include "../core/bsdf.h"
#include"../core/kulla_conty.h"

///\brief 粗糙的塑料材质派生类
class RoughPlastic : public Bsdf
{
public:
    /**
     * @brief 粗糙的塑料材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param eta 相对折射率
     * @param diffuse_reflectance 漫反射系数
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param distri 微表面分布类型
     * @param alpha 表面的粗糙程度
     * @param nonlinear 是否考虑因内部散射而引起的非线性色移
     * @param kulla_conty_lut Kulla-Conty 补偿散射能量查找表
     * @param albedo_avg 平均反照率
     */
    __device__ RoughPlastic(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                            vec3 eta, Texture *diffuse_reflectance, Texture *specular_reflectance,
                            MicrofacetDistribType distri, Texture *alpha, bool nonlinear,
                            float *kulla_conty_lut, float albedo_avg)
        : Bsdf(idx, kRoughPlastic, twosided, bump_map, opacity_map),
          eta_inv_d_(1.0 / eta.x), diffuse_reflectance_(diffuse_reflectance),
          specular_reflectance_(specular_reflectance), distri_(distri), alpha_(alpha),
          nonlinear_(nonlinear), fdr_int_(AverageFresnel(eta.x)), albedo_avg_(-1),
          kulla_conty_lut_(nullptr), f_add_(vec3(0))
    {
        if (albedo_avg < 0)
            return;
        albedo_avg_ = albedo_avg;
        kulla_conty_lut_ = kulla_conty_lut;

        f_add_ = vec3(fdr_int_ * fdr_int_ * albedo_avg_ / (1.0 - fdr_int_ * (1.0 - albedo_avg_)));
    }

    __device__ ~RoughPlastic()
    {
        if (kulla_conty_lut_)
        {
            delete[] kulla_conty_lut_;
            kulla_conty_lut_ = nullptr;
        }
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const override
    {
        Float alpha = alpha_ ? alpha_->Color(rec.texcoord).x : 0.1,
              specular_sampling_weight = SpecularSamplingWeight(rec.texcoord),
              kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_d_);
        Float pdf_specular = kr_o * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr_o) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);

        auto h = vec3(0);
        Float D = 0;
        if (sample.z < pdf_specular)
        {
            SampleNormDistrib(distri_, alpha, alpha, rec.normal, sample, h, D);
            rec.wi = -Reflect(-rec.wo, h);
            if (myvec::dot(rec.wi, rec.normal) >= 0)
                return;
        }
        else
        {
            auto wi_local = vec3(0);
            Float pdf = 0;
            HemisCos(sample.x, sample.y, wi_local, pdf);
            rec.wi = -ToWorld(wi_local, rec.normal);
            h = myvec::normalize(-rec.wi + rec.wo);
            D = PdfNormDistrib(distri_, alpha, alpha, rec.normal, h);
        }
        Float kr_i = Fresnel(rec.wi, rec.normal, eta_inv_d_);
        pdf_specular = kr_i * specular_sampling_weight,
        pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(ToLocal(rec.wo, rec.normal));
        if (D > kEpsilon)
            rec.pdf += pdf_specular * D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
        if (rec.pdf < kEpsilonL)
            return;

        vec3 diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(rec.texcoord) : vec3(0.5);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_int_);
        rec.attenuation *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        Float cos_theta_i = myvec::dot(-rec.wi, rec.normal);
        if (D > kEpsilon)
        {
            Float cos_theta_o = myvec::dot(rec.wo, rec.normal),
                  F = Fresnel(rec.wi, h, eta_inv_d_),
                  G = SmithG1(distri_, alpha, alpha, -rec.wi, rec.normal, h) *
                      SmithG1(distri_, alpha, alpha, rec.wo, rec.normal, h);
            auto value = vec3(F * D * G / (4.0 * abs(cos_theta_i * cos_theta_o)));
            if (albedo_avg_ > 0)
                value += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            if (specular_reflectance_)
                value *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += value;
        }
        rec.attenuation *= cos_theta_i;
        rec.valid = true;
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(rec.wo, rec.normal))
            return;

        Float alpha = alpha_ ? alpha_->Color(rec.texcoord).x : 0.1,
              kr_i = Fresnel(rec.wi, rec.normal, eta_inv_d_),
              specular_sampling_weight = SpecularSamplingWeight(rec.texcoord);
        Float pdf_specular = kr_i * specular_sampling_weight,
              pdf_diffuse = (1.0 - kr_i) * (1.0 - specular_sampling_weight);
        pdf_specular = pdf_specular / (pdf_specular + pdf_diffuse);
        rec.pdf = (1.0 - pdf_specular) * PdfHemisCos(ToLocal(rec.wo, rec.normal));
        vec3 h = myvec::normalize(-rec.wi + rec.wo);
        Float D = PdfNormDistrib(distri_, alpha, alpha, rec.normal, h);
        if (D > kEpsilon)
            rec.pdf += pdf_specular * D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
        if (rec.pdf < kEpsilonL)
            return;
        rec.valid = true;

        vec3 diffuse_reflectance = diffuse_reflectance_ ? diffuse_reflectance_->Color(rec.texcoord) : vec3(0.5);
        if (nonlinear_)
            rec.attenuation = diffuse_reflectance / (vec3(1) - diffuse_reflectance * fdr_int_);
        else
            rec.attenuation = diffuse_reflectance / (1.0 - fdr_int_);
        Float kr_o = Fresnel(-rec.wo, rec.normal, eta_inv_d_);
        rec.attenuation *= eta_inv_d_ * eta_inv_d_ * (1.0 - kr_i) * (1.0 - kr_o) * kPiInv;
        Float cos_theta_i = myvec::dot(-rec.wi, rec.normal);
        if (D > kEpsilon)
        {
            Float cos_theta_o = myvec::dot(rec.wo, rec.normal),
                  F = Fresnel(rec.wi, h, eta_inv_d_),
                  G = SmithG1(distri_, alpha, alpha, -rec.wi, rec.normal, h) *
                      SmithG1(distri_, alpha, alpha, rec.wo, rec.normal, h);
            auto value = vec3(F * D * G / (4.0 * abs(cos_theta_i * cos_theta_o)));
            if (albedo_avg_ > 0)
                value += EvalMultipleScatter(cos_theta_i, cos_theta_o);
            if (specular_reflectance_)
                value *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation += value;
        }
        rec.attenuation *= cos_theta_i;
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || diffuse_reflectance_ && diffuse_reflectance_->Varying() ||
               specular_reflectance_ && specular_reflectance_->Varying() || alpha_ && alpha_->Varying();
    }

    ///\return 给定点是否透明
    __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const override
    {
        if (Bsdf::Transparent(texcoord, sample))
            return true;

        if (diffuse_reflectance_)
        {
            if (diffuse_reflectance_->IsBitmap() &&
                diffuse_reflectance_->Transparent(texcoord, sample.y))
                return true;
        }
        return false;
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

    ///\brief 给定光线出射方向与法线方向夹角的余弦，获取平均反照率
    __device__ Float GetAlbedo(Float cos_theta) const
    {
        Float offset = cos_theta * kAlbedoResolution;
        auto idx = static_cast<int>(offset);
        if (idx >= kAlbedoResolution - 1)
            return kulla_conty_lut_[kAlbedoResolution - 1];
        else
            return (1.0 - (offset - idx)) * kulla_conty_lut_[idx] +
                   (offset - idx) * kulla_conty_lut_[idx + 1];
    }

    ///\brief 补偿多次散射后又射出的光能
    __device__ vec3 EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o) const
    {
        Float albedo_i = GetAlbedo(abs(cos_theta_i)),
              albedo_o = GetAlbedo(abs(cos_theta_o)),
              f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
        return f_ms * f_add_;
    }

    Float eta_inv_d_;               //外部折射率与介质折射率之比
    Texture *diffuse_reflectance_;  //漫反射系数
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    MicrofacetDistribType distri_;  //微表面分布类型
    Texture *alpha_;                //表面的粗糙程度
    bool nonlinear_;                //是否考虑因内部散射而引起的非线性色移
    Float fdr_int_;                 //漫反射菲涅尔项平均值
    float albedo_avg_;              //平均反照率
    float *kulla_conty_lut_;        // Kulla-Conty 补偿散射能量查找表
    vec3 f_add_;                    // Kulla-Conty 补偿散射能量系数
};

__device__ inline void InitRoughPlastic(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
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

    Texture *alpha = nullptr;
    if (bsdf_info_list[m_idx].alpha_u_idx != kUintMax)
        alpha = texture_list + bsdf_info_list[m_idx].alpha_u_idx;

    auto albedo_avg = static_cast<float>(-1);
    float *kulla_conty_lut = nullptr;
    CreateCosinAlbedoTexture(bsdf_info_list[m_idx].distri, alpha, alpha,
                             kulla_conty_lut, albedo_avg);

    bsdf_list[m_idx] = new RoughPlastic(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                            bsdf_info_list[m_idx].eta, diffuse_reflectance, specular_reflectance,
                                            bsdf_info_list[m_idx].distri, alpha, bsdf_info_list[m_idx].nonlinear,
                                            kulla_conty_lut, albedo_avg);
}