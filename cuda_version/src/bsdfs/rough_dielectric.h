#pragma once

#include "../core/bsdf_base.h"

///\brief 粗糙的电介质派生类
class RoughDielectric : public Bsdf
{
public:
    /**
     * @brief 粗糙的电介质材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param eta 相对折射率
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param distri 微表面分布类型
     * @param alpha_u 沿切线（tangent）方向的粗糙度
     * @param alpha_v 沿副切线（bitangent）方向的粗糙度
     * @param kulla_conty_lut Kulla-Conty 补偿散射能量查找表
     * @param albedo_avg 平均反照率
     */
    __device__ RoughDielectric(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                               vec3 eta, Texture *specular_reflectance, Texture *specular_transmittance,
                               MicrofacetDistribType distri, Texture *alpha_u, Texture *alpha_v,
                               float *kulla_conty_lut, float albedo_avg)
        : Bsdf(idx, kRoughDielectric, twosided, bump_map, opacity_map),
          eta_d_(eta.x), eta_inv_d_(1.0 / eta.x), specular_reflectance_(specular_reflectance),
          specular_transmittance_(specular_transmittance), distri_(distri), alpha_u_(alpha_u),
          alpha_v_(alpha_v), albedo_avg_(-1), kulla_conty_lut_(nullptr), f_add_(vec3(0)),
          f_add_inv_(vec3(0)), ratio_t_(0), ratio_t_inv_(0)
    {
        if (albedo_avg < 0)
            return;
        albedo_avg_ = albedo_avg;
        kulla_conty_lut_ = kulla_conty_lut;

        Float F_avg = AverageFresnel(eta_d_);
        f_add_ = vec3(F_avg * albedo_avg / (1.0 - F_avg * (1.0 - albedo_avg)));

        Float F_avg_inv = AverageFresnel(eta_inv_d_);
        f_add_inv_ = vec3(F_avg_inv * albedo_avg / (1.0 - F_avg_inv * (1.0 - albedo_avg)));

        ratio_t_ = (1.0 - F_avg) * (1.0 - F_avg_inv) * eta_d_ * eta_d_ /
                   ((1.0 - F_avg) + (1.0 - F_avg_inv) * eta_d_ * eta_d_);

        ratio_t_inv_ = (1.0 - F_avg_inv) * (1.0 - F_avg) * eta_inv_d_ * eta_inv_d_ /
                       ((1.0 - F_avg_inv) + (1.0 - F_avg) * eta_inv_d_ * eta_inv_d_);
    }

    __device__ ~RoughDielectric()
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
        Float eta = rec.inside == kTrue ? eta_inv_d_ : eta_d_,   //相对折射率，即光线透射侧介质折射率与入射侧介质折射率之比
            eta_inv = rec.inside == kTrue ? eta_d_ : eta_inv_d_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            ratio_t = rec.inside == kTrue ? ratio_t_inv_ : ratio_t_,
              ratio_t_inv = rec.inside == kTrue ? ratio_t_ : ratio_t_inv_,
              alpha_u = alpha_u_ ? alpha_u_->Color(rec.texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(rec.texcoord).x : 0.1;

        // Walter 等人在《Microfacet Models for Refraction through Rough Surfaces》中提到的技巧，略微缩放粗糙度，以减少重要性采样权重。
        Float scale = 1.2 - 0.2 * sqrt(abs(myvec::dot(-rec.wo, rec.normal)));
        alpha_u *= scale;
        alpha_v *= scale;

        auto h = vec3(0);
        Float D = 0;
        SampleNormDistrib(distri_, alpha_u, alpha_v, rec.normal, sample, h, D);
        if (D < kEpsilonPdf)
            return;

        Float F = Fresnel(-rec.wo, h, eta_inv);
        if (sample.z < F)
        {
            rec.wi = -Reflect(-rec.wo, h);
            Float cos_theta_i = myvec::dot(-rec.wi, rec.normal);
            if (cos_theta_i < kEpsilon)
                return;

            rec.pdf = F * D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
            if (rec.pdf < kEpsilonPdf)
                return;

            Float G = SmithG1(distri_, alpha_u, alpha_v, -rec.wi, rec.normal, h) *
                      SmithG1(distri_, alpha_u, alpha_v, rec.wo, rec.normal, h),
                  cos_theta_o = myvec::dot(rec.wo, rec.normal);
            rec.attenuation = vec3(F * D * G / (4.0 * abs(cos_theta_i * cos_theta_o)));
            if (albedo_avg_ > 0)
                rec.attenuation += (1 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
            rec.attenuation *= cos_theta_i;
        }
        else
        {
            rec.wi = -Refract(-rec.wo, h, eta_inv);
            Float cos_theta_i = myvec::dot(rec.wi, rec.normal);
            if (cos_theta_i <= kEpsilon)
                return;
            {
                rec.normal = -rec.normal;
                rec.inside = !rec.inside;
                eta_inv = eta;
                h = -h;
                ratio_t = ratio_t_inv;
            }

            F = Fresnel(rec.wi, h, eta_inv);
            Float cos_i_h = myvec::dot(-rec.wi, h),
                  cos_o_h = myvec::dot(rec.wo, h);
            rec.pdf = (1.0 - F) * D * abs(cos_o_h / pow(eta_inv * cos_i_h + cos_o_h, 2));
            if (rec.pdf < kEpsilonPdf)
                return;

            Float G = SmithG1(distri_, alpha_u, alpha_v, -rec.wi, rec.normal, h) *
                      SmithG1(distri_, alpha_u, alpha_v, rec.wo, rec.normal, h),
                  cos_theta_o = myvec::dot(rec.wo, rec.normal);
            rec.attenuation = vec3(abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                       (cos_theta_i * cos_theta_o * pow(eta_inv * cos_i_h + cos_o_h, 2))));
            if (albedo_avg_ > 0)
                rec.attenuation += ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            rec.attenuation *= cos_theta_i;
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= eta_inv * eta_inv;
        }
        rec.valid = true;
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        Float eta_inv = rec.inside ? eta_d_ : eta_inv_d_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            alpha_u = alpha_u_ ? alpha_u_->Color(rec.texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(rec.texcoord).x : 0.1,
              cos_theta_o = myvec::dot(rec.wo, rec.normal);

        auto h = vec3(0);
        bool relfect = cos_theta_o > 0;
        if (relfect)
            h = myvec::normalize(-rec.wi + rec.wo);
        else
        {
            h = myvec::normalize(-eta_inv * rec.wi + rec.wo);
            if (NotSameHemis(h, rec.normal))
                h = -h;
        }

        Float D = PdfNormDistrib(distri_, alpha_u, alpha_v, rec.normal, h),
              F = Fresnel(rec.wi, h, eta_inv);
        if (relfect)
            rec.pdf = F * D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
        else
            rec.pdf = (1.0 - F) * D * abs(myvec::dot(rec.wo, h) / pow(eta_inv * myvec::dot(-rec.wi, h) + myvec::dot(rec.wo, h), 2));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        Float ratio_t = rec.inside ? ratio_t_inv_ : ratio_t_,
              cos_theta_i = myvec::dot(-rec.wi, rec.normal);
        Float G = SmithG1(distri_, alpha_u, alpha_v, -rec.wi, rec.normal, h) *
                  SmithG1(distri_, alpha_u, alpha_v, rec.wo, rec.normal, h);
        if (relfect)
        {
            rec.attenuation = vec3(F * D * G / (4.0 * abs(cos_theta_i * cos_theta_o)));
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
            if (albedo_avg_ > 0)
                rec.attenuation += (1 - ratio_t) * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
            rec.attenuation *= cos_theta_i;
        }
        else
        {
            Float cos_i_h = myvec::dot(-rec.wi, h),
                  cos_o_h = myvec::dot(rec.wo, h);
            rec.attenuation = vec3(abs(cos_i_h * cos_o_h * (1.0 - F) * G * D /
                                       (cos_theta_i * cos_theta_o * pow(eta_inv * cos_i_h + cos_o_h, 2))));
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            if (albedo_avg_ > 0)
                rec.attenuation += ratio_t * EvalMultipleScatter(cos_theta_i, cos_theta_o, rec.inside);
            rec.attenuation *= cos_theta_i;
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= eta_inv * eta_inv;
        }
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && specular_reflectance_->Varying() ||
               specular_transmittance_ && specular_transmittance_->Varying() || alpha_u_ && alpha_u_->Varying() ||
               alpha_v_ && alpha_v_->Varying();
    }

private:
    ///\brief 给定光线出射方向与法线方向夹角的余弦，获取平均反照率
    __device__ Float GetAlbedo(Float cos_theta) const
    {
        Float offset = cos_theta * kAlbedoResolution;
        auto idx = static_cast<int>(offset);
        if (idx >= kAlbedoResolution - 1)
            return kulla_conty_lut_[kAlbedoResolution - 1];
        else
            return (1.0 - (offset - idx)) * kulla_conty_lut_[idx] + (offset - idx) * kulla_conty_lut_[idx + 1];
    }

    ///\brief 补偿多次散射后又射出的光能
    __device__ vec3 EvalMultipleScatter(Float cos_theta_i, Float cos_theta_o, int inside) const
    {
        vec3 f_add = (inside == kTrue) ? f_add_inv_ : f_add_;
        Float albedo_i = GetAlbedo(abs(cos_theta_i)),
              albedo_o = GetAlbedo(abs(cos_theta_o));
        vec3 f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
        return f_ms * f_add;
    }

    Float eta_d_;                     //介质折射率与外部折射率之比
    Float eta_inv_d_;                 //外部折射率与介质折射率之比
    Texture *specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    Texture *specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    MicrofacetDistribType distri_;    //微表面分布类型
    Texture *alpha_u_;                //沿切线（tangent）方向的粗糙度
    Texture *alpha_v_;                //沿副切线（bitangent）方向的粗糙度
    float albedo_avg_;                //平均反照率
    float *kulla_conty_lut_;          // Kulla-Conty 补偿散射能量查找表
    vec3 f_add_;                      // 光线从外部射入介质，Kulla-Conty 补偿散射能量系数
    vec3 f_add_inv_;                  // 光线从介质内部射出，Kulla-Conty 补偿散射能量系数
    Float ratio_t_;                   // 光线从外部射入介质，Kulla-Conty 补偿散射能量中折射的比例
    Float ratio_t_inv_;               // 光线从介质内部射出，Kulla-Conty 补偿散射能量中折射的比例
};

__device__ inline void InitRoughDielectric(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
                                           Bsdf **&bsdf_list)
{
    Texture *bump_map = nullptr;
    if (bsdf_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + bsdf_info_list[m_idx].bump_map_idx;

    Texture *opacity_map = nullptr;
    if (bsdf_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + bsdf_info_list[m_idx].opacity_idx;

    Texture *specular_reflectance = nullptr;
    if (bsdf_info_list[m_idx].specular_reflectance_idx != kUintMax)
        specular_reflectance = texture_list + bsdf_info_list[m_idx].specular_reflectance_idx;

    Texture *specular_transmittance = nullptr;
    if (bsdf_info_list[m_idx].specular_transmittance_idx != kUintMax)
        specular_transmittance = texture_list + bsdf_info_list[m_idx].specular_transmittance_idx;

    Texture *alpha_u = nullptr;
    if (bsdf_info_list[m_idx].alpha_u_idx != kUintMax)
        alpha_u = texture_list + bsdf_info_list[m_idx].alpha_u_idx;

    Texture *alpha_v = nullptr;
    if (bsdf_info_list[m_idx].alpha_v_idx != kUintMax)
        alpha_v = texture_list + bsdf_info_list[m_idx].alpha_v_idx;

    auto albedo_avg = static_cast<float>(-1);
    float *kulla_conty_lut = nullptr;
    CreateCosinAlbedoTexture(bsdf_info_list[m_idx].distri, alpha_u, alpha_v,
                             kulla_conty_lut, albedo_avg);

    bsdf_list[m_idx] = new RoughDielectric(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                               bsdf_info_list[m_idx].eta, specular_reflectance, specular_transmittance,
                                               bsdf_info_list[m_idx].distri, alpha_u, alpha_v, kulla_conty_lut,
                                               albedo_avg);
}
