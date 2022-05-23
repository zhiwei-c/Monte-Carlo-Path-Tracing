#pragma once

#include "../core/material_base.h"

///\brief 粗糙的导体材质派生类
class RoughConductor : public Material
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
    __device__ RoughConductor(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                              bool mirror, vec3 eta, vec3 k, Texture *specular_reflectance,
                              MicrofacetDistribType distri, Texture *alpha_u, Texture *alpha_v,
                              float *kulla_conty_lut, float albedo_avg)
        : Material(idx, kRoughConductor, twosided, bump_map, opacity_map),
          mirror_(mirror), eta_(eta), k_(k), specular_reflectance_(specular_reflectance),
          distri_(distri), alpha_u_(alpha_u), alpha_v_(alpha_v), albedo_avg_(-1),
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
    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const override
    {
        Float alpha_u = alpha_u_ ? alpha_u_->Color(bs.texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(bs.texcoord).x : 0.1;

        auto h = vec3(0);
        Float D = 0;
        SampleNormDistrib(distri_, alpha_u, alpha_v, bs.normal, sample, h, D);

        bs.wi = -Reflect(-bs.wo, h);
        Float cos_i_n = myvec::dot(bs.wi, bs.normal);
        if (cos_i_n >= 0)
            return;

        bs.pdf = D * abs(1.0 / (4.0 * myvec::dot(bs.wo, h)));
        if (bs.pdf < kEpsilonPdf)
            return;

        vec3 F = mirror_ ? vec3(1) : FresnelConductor(bs.wi, h, eta_, k_);
        Float G = SmithG1(distri_, alpha_u, alpha_v, -bs.wi, bs.normal, h) *
                  SmithG1(distri_, alpha_u, alpha_v, bs.wo, bs.normal, h),
              cos_o_n = myvec::dot(bs.wo, bs.normal);
        bs.attenuation = F * static_cast<Float>(D * G / abs(4.0 * -cos_i_n * cos_o_n));
        if (albedo_avg_ > 0)
            bs.attenuation += EvalMultipleScatter(cos_i_n, cos_o_n);
        if (specular_reflectance_)
            bs.attenuation *= specular_reflectance_->Color(bs.texcoord);

        bs.valid = true;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        Float alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1,
              cos_i_n = abs(myvec::dot(wi, normal)),
              cos_o_n = abs(myvec::dot(wo, normal));
        vec3 h = myvec::normalize(-wi + wo),
             F = mirror_ ? vec3(1) : FresnelConductor(wi, h, eta_, k_);
        Float D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h),
              G = SmithG1(distri_, alpha_u, alpha_v, -wi, normal, h) *
                  SmithG1(distri_, alpha_u, alpha_v, wo, normal, h);
        vec3 albedo = F * static_cast<Float>(D * G / (4.0 * cos_i_n * cos_o_n));
        if (specular_reflectance_)
            albedo *= specular_reflectance_->Color(texcoord);
        if (albedo_avg_ > 0)
            albedo += EvalMultipleScatter(cos_i_n, cos_o_n);
        return albedo;
    }

    ///\return 根据光线入射方向和法线方向计算的，光线从给定出射方向射出的概率
    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(wo, normal))
            return 0;
        vec3 h = myvec::normalize(-wi + wo);
        Float alpha_u = alpha_u_ ? alpha_u_->Color(texcoord).x : 0.1,
              alpha_v = alpha_v_ ? alpha_v_->Color(texcoord).x : 0.1,
              D = PdfNormDistrib(distri_, alpha_u, alpha_v, normal, h);
        if (D < kEpsilonL)
            return 0;
        else
            return D * abs(1.0 / (4.0 * myvec::dot(wo, h)));
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               specular_reflectance_ && specular_reflectance_->Varying() ||
               alpha_u_ && alpha_u_->Varying() ||
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
            return (1.0 - (offset - idx)) * kulla_conty_lut_[idx] +
                   (offset - idx) * kulla_conty_lut_[idx + 1];
    }

    ///\brief 补偿多次散射后又射出的光能
    __device__ vec3 EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
    {
        Float albedo_i = GetAlbedo(abs(cos_i_n)),
              albedo_o = GetAlbedo(abs(cos_o_n)),
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

__device__ inline void InitRoughConductor(uint m_idx,
                                          MaterialInfo *material_info_list,
                                          Texture *texture_list,
                                          Material **&material_list)
{
    Texture *bump_map = nullptr;
    if (material_info_list[m_idx].bump_map_idx != kUintMax)
        bump_map = texture_list + material_info_list[m_idx].bump_map_idx;

    Texture *opacity_map = nullptr;
    if (material_info_list[m_idx].opacity_idx != kUintMax)
        opacity_map = texture_list + material_info_list[m_idx].opacity_idx;

    Texture *specular_reflectance = nullptr;
    if (material_info_list[m_idx].specular_reflectance_idx != kUintMax)
        specular_reflectance = texture_list + material_info_list[m_idx].specular_reflectance_idx;

    Texture *alpha_u = nullptr;
    if (material_info_list[m_idx].alpha_u_idx != kUintMax)
        alpha_u = texture_list + material_info_list[m_idx].alpha_u_idx;

    Texture *alpha_v = nullptr;
    if (material_info_list[m_idx].alpha_v_idx != kUintMax)
        alpha_v = texture_list + material_info_list[m_idx].alpha_v_idx;

    auto albedo_avg = static_cast<float>(-1);
    float *kulla_conty_lut = nullptr;
    CreateCosinAlbedoTexture(material_info_list[m_idx].distri, alpha_u, alpha_v,
                             kulla_conty_lut, albedo_avg);

    material_list[m_idx] = new RoughConductor(m_idx,
                                              material_info_list[m_idx].twosided,
                                              bump_map,
                                              opacity_map,
                                              material_info_list[m_idx].mirror,
                                              material_info_list[m_idx].eta,
                                              material_info_list[m_idx].k,
                                              specular_reflectance,
                                              material_info_list[m_idx].distri,
                                              alpha_u,
                                              alpha_v,
                                              kulla_conty_lut,
                                              albedo_avg);
}