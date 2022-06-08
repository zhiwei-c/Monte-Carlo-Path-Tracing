#pragma once

#include "../core/bsdf.h"
#include "../core/kulla_conty.h"

///\brief 粗糙的导体材质派生类
class RoughConductor : public Bsdf
{
public:
    /**
     * @brief 粗糙的导体材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param mirror 是否是镜面
     * @param eta 复数折射率的实部
     * @param k 复数折射率的虚部（消光系数）
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param distri 微表面分布类型
     * @param alpha_u 沿切线（tangent）方向的粗糙度
     * @param alpha_v 沿副切线（bitangent）方向的粗糙度
     * @param kulla_conty_lut Kulla-Conty 补偿散射能量查找表
     * @param albedo_avg 平均反照率
     */
    __device__ RoughConductor(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map, bool mirror, vec3 eta, vec3 k,
                              Texture *specular_reflectance, MicrofacetDistribType distri, Texture *alpha_u, Texture *alpha_v,
                              float *kulla_conty_lut, float albedo_avg)
        : Bsdf(idx, kRoughConductor, twosided, bump_map, opacity_map), mirror_(mirror), eta_(eta), k_(k),
          specular_reflectance_(specular_reflectance), distri_(distri), alpha_u_(alpha_u), alpha_v_(alpha_v), albedo_avg_(-1),
          kulla_conty_lut_(nullptr), f_add_(vec3(0))
    {
        if (albedo_avg < 0)
            return;
        albedo_avg_ = albedo_avg;
        kulla_conty_lut_ = kulla_conty_lut;

        auto reflectivity = vec3(0),
             edgetint = vec3(0);
        IorToReflectivityEdgetint(eta_, k_, reflectivity, edgetint);

        vec3 F_avg = AverageFresnelConductor(reflectivity, edgetint);
        f_add_ = F_avg * F_avg * albedo_avg / (vec3(1) - F_avg * (1.0 - albedo_avg));
    }

    __device__ ~RoughConductor()
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
        Float alpha_u = alpha_u_ ? alpha_u_->Color(rec.texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(rec.texcoord).x : 0.1;

        auto h = vec3(0);
        Float D = 0;
        SampleNormDistrib(distri_, alpha_u, alpha_v, rec.normal, sample, h, D);

        rec.wi = -Reflect(-rec.wo, h);
        Float cos_theta_i = myvec::dot(-rec.wi, rec.normal);
        if (cos_theta_i < kEpsilon)
            return;

        rec.pdf = D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        vec3 F = mirror_ ? vec3(1) : FresnelConductor(rec.wi, h, eta_, k_);
        Float G = SmithG1(distri_, alpha_u, alpha_v, -rec.wi, rec.normal, h) *
                  SmithG1(distri_, alpha_u, alpha_v, rec.wo, rec.normal, h),
              cos_theta_o = myvec::dot(rec.wo, rec.normal);
        rec.attenuation = F * static_cast<Float>(D * G / abs(4.0 * cos_theta_i * cos_theta_o));
        if (albedo_avg_ > 0)
            rec.attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
        if (specular_reflectance_)
            rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        rec.attenuation *= cos_theta_i;
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(rec.wo, rec.normal))
            return;

        vec3 h = myvec::normalize(-rec.wi + rec.wo);
        Float alpha_u = alpha_u_ ? alpha_u_->Color(rec.texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(rec.texcoord).x : 0.1,
              D = PdfNormDistrib(distri_, alpha_u, alpha_v, rec.normal, h);
        rec.pdf = D * abs(1.0 / (4.0 * myvec::dot(rec.wo, h)));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        Float cos_theta_i = abs(myvec::dot(rec.wi, rec.normal)),
              cos_theta_o = abs(myvec::dot(rec.wo, rec.normal));
        vec3 F = mirror_ ? vec3(1) : FresnelConductor(rec.wi, h, eta_, k_);
        Float G = SmithG1(distri_, alpha_u, alpha_v, -rec.wi, rec.normal, h) *
                  SmithG1(distri_, alpha_u, alpha_v, rec.wo, rec.normal, h);
        rec.attenuation = F * static_cast<Float>(D * G / (4.0 * cos_theta_i * cos_theta_o));
        if (specular_reflectance_)
            rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        if (albedo_avg_ > 0)
            rec.attenuation += EvalMultipleScatter(cos_theta_i, cos_theta_o);
        rec.attenuation *= cos_theta_i;
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && specular_reflectance_->Varying() ||
               alpha_u_ && alpha_u_->Varying() || alpha_v_ && alpha_v_->Varying();
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

    bool mirror_;                   //是否是镜面
    vec3 eta_;                      //复数折射率的实部
    vec3 k_;                        //复数折射率的虚部（消光系数）
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    MicrofacetDistribType distri_;  //微表面分布类型
    Texture *alpha_u_;              //沿切线（tangent）方向的粗糙度
    Texture *alpha_v_;              //沿副切线（bitangent）方向的粗糙度
    float albedo_avg_;              //平均反照率
    float *kulla_conty_lut_;        // Kulla-Conty 补偿散射能量查找表
    vec3 f_add_;                    // Kulla-Conty 补偿散射能量系数
};

__device__ inline void InitRoughConductor(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
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

    bsdf_list[m_idx] = new RoughConductor(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                          bsdf_info_list[m_idx].mirror, bsdf_info_list[m_idx].eta,
                                          bsdf_info_list[m_idx].k, specular_reflectance,
                                          bsdf_info_list[m_idx].distri, alpha_u, alpha_v, kulla_conty_lut,
                                          albedo_avg);
}