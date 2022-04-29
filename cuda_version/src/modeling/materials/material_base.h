#pragma once

#include "material_info.h"

struct BsdfSampling
{
    bool valid;
    int inside;       //表面法线方向是否朝向表面内侧
    Float pdf;        //光线从该方向入射的概率
    vec2 texcoord;    //表面纹理坐标，可选
    vec3 wi;          //光线入射方向
    vec3 wo;          //光线出射方向
    vec3 normal;      //表面法线方向
    vec3 attenuation; // BSDF 光能衰减系数

    __device__ BsdfSampling()
        : valid(kFalse), inside(0), pdf(0), texcoord(vec2(0)), wi(vec3(0)), wo(vec3(0)), normal(vec3(0)), attenuation(vec3(0)) {}
};

class Material
{
public:
    __device__ Material()
        : idx_(kUintMax),
          type_(kNoneMaterial),
          twosided_(false),
          bump_map_(nullptr),
          opacity_map_(nullptr),
          radiance_(nullptr),
          mirror_(true),
          eta_d_(1),
          eta_inv_d_(1),
          eta_(vec3(1)),
          k_(vec3(0)),
          diffuse_reflectance_(nullptr),
          specular_reflectance_(nullptr),
          specular_transmittance_(nullptr),
          distri_(kNoneDistrib),
          alpha_u_(nullptr),
          alpha_v_(nullptr),
          fdr_int_(0),
          nonlinear_(false),
          pre_(nullptr),
          next_(nullptr),
          kulla_conty_table_(nullptr),
          albedo_avg_(-1),
          f_add_(vec3(0)),
          f_add_inv_(vec3(0)),
          ratio_t_(0),
          ratio_t_inv_(0) {}

    __device__ ~Material()
    {
        if (kulla_conty_table_)
        {
            delete[] kulla_conty_table_;
            kulla_conty_table_ = nullptr;
        }
    }

    /**
     * \brief 根据光线出射方向和表面法线方向，抽样光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 由 vec3 类型和 BsdfSamplingType 类型构成的 pair，分别代表抽样所得光线入射方向，和入射光线与出射光线之间的关系
     */
    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const;

    /**
     * \brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
     * \param wi 光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return BSDF 光能衰减系数
     */
    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    /**
     * \brief 根据光线入射方向、出射方向和法线方向，计算光线因从入射方向入射，而从出射方向出射的概率
     * \param wi 光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 光线因从入射方向入射，而从出射方向出射的概率
     */
    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    ///\return 是否发光
    __device__ bool HasEmission() const;

    __device__ bool twosided() const { return twosided_; }

    __device__ bool HarshLobe() const { return type_ == kConductor || type_ == kDielectric || type_ == kThinDielectric; }

    ///\return 辐射亮度
    __device__ vec3 radiance() const;

    __device__ void InitAreaLight(bool twosided, Texture *radiance);

    __device__ void InitDiffuse(bool twosided, Texture *bump_map, Texture *opacity_map, Texture *diffuse_reflectance);

    __device__ void InitDielectric(bool twosided,
                                   Texture *bump_map,
                                   Texture *opacity_map,
                                   vec3 eta,
                                   Texture *specular_reflectance,
                                   Texture *specular_transmittance);

    __device__ void InitRoughDielectric(bool twosided,
                                        Texture *bump_map,
                                        Texture *opacity_map,
                                        vec3 eta,
                                        Texture *specular_reflectance,
                                        Texture *specular_transmittance,
                                        MicrofacetDistribType distri,
                                        Texture *alpha_u,
                                        Texture *alpha_v,
                                        float *kulla_conty_table,
                                        float albedo_avg);

    __device__ void InitThinDielectric(bool twosided,
                                       Texture *bump_map,
                                       Texture *opacity_map,
                                       vec3 eta,
                                       Texture *specular_reflectance,
                                       Texture *specular_transmittance);

    __device__ void InitConductor(bool twosided,
                                  Texture *bump_map,
                                  Texture *opacity_map,
                                  bool mirror,
                                  vec3 eta,
                                  vec3 k,
                                  Texture *specular_reflectance);

    __device__ void InitRoughConductor(bool twosided,
                                       Texture *bump_map,
                                       Texture *opacity_map,
                                       bool mirror,
                                       vec3 eta,
                                       vec3 k,
                                       Texture *specular_reflectance,
                                       MicrofacetDistribType distri,
                                       Texture *alpha_u,
                                       Texture *alpha_v,
                                       float *kulla_conty_table,
                                       float albedo_avg);

    __device__ void InitPlastic(bool twosided,
                                Texture *bump_map,
                                Texture *opacity_map,
                                vec3 eta,
                                Texture *diffuse_reflectance,
                                Texture *specular_reflectance,
                                bool nonlinear);

    __device__ void InitRoughPlastic(bool twosided,
                                     Texture *bump_map,
                                     Texture *opacity_map,
                                     vec3 eta,
                                     Texture *diffuse_reflectance,
                                     Texture *specular_reflectance,
                                     MicrofacetDistribType distri,
                                     Texture *alpha,
                                     bool nonlinear,
                                     float *kulla_conty_table,
                                     float albedo_avg);

    __device__ void SetOtherInfo(uint idx, Material *pre, Material *next)
    {
        idx_ = idx;
        pre_ = pre;
        next_ = next;
    }

    __device__ bool BumpMapping() const { return bump_map_ != nullptr; }

    __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const
    {
        if (opacity_map_)
        {
            if (opacity_map_->Transparent(texcoord, sample.x))
                return true;
        }
        if (diffuse_reflectance_)
        {
            if (diffuse_reflectance_->type() == kBitmap &&
                diffuse_reflectance_->Transparent(texcoord, sample.y))
                return true;
        }
        return false;
    }

    __device__ vec3 PerturbNormal(const vec3 &normal, const vec3 &tangent, const vec3 &bitangent, const vec2 &texcoord) const
    {
        auto TBN = gmat3(gvec3(tangent.x, tangent.y, tangent.z),
                         gvec3(bitangent.x, bitangent.y, bitangent.z),
                         gvec3(normal.x, normal.y, normal.z));
        auto gradient = bump_map_->Gradient(texcoord);
        auto normal_pertubed_local = glm::normalize(gvec3(-gradient.x, -gradient.y, 1));
        auto normal_pertubed = glm::normalize(TBN * normal_pertubed_local);
        return vec3(normal_pertubed.x, normal_pertubed.y, normal_pertubed.z);
    }

private:
    uint idx_;
    Material *pre_;
    Material *next_;
    MaterialType type_;
    bool twosided_;
    Texture *bump_map_;
    Texture *opacity_map_;
    bool mirror_;
    Float eta_d_;
    Float eta_inv_d_;
    vec3 eta_;
    vec3 k_;
    Texture *radiance_;
    Texture *diffuse_reflectance_;
    Texture *specular_reflectance_;
    Texture *specular_transmittance_;
    MicrofacetDistribType distri_;
    Texture *alpha_u_;
    Texture *alpha_v_;
    bool nonlinear_;
    float albedo_avg_;
    float *kulla_conty_table_;
    vec3 f_add_;
    vec3 f_add_inv_;
    Float ratio_t_;
    Float ratio_t_inv_;
    Float fdr_int_;

    ///\brief 获取给定点抽样镜面反射的权重
    __device__ Float SpecularSamplingWeight(const vec2 &texcoord) const;

    __device__ Float GetAlbedo(Float cos_theta) const;

    ///\brief 补偿多次散射后又射出的光能
    __device__ vec3 EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const;

    ///\brief 补偿多次散射后又射出的光能
    __device__ vec3 EvalMultipleScatter(Float cos_i_n, Float cos_o_n, int inside) const;

    __device__ void SampleDiffuse(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalDiffuse(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfDiffuse(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleDielectric(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleRoughDielectric(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalRoughDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfRoughDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleThinDielectric(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalThinDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfThinDielectric(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleConductor(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleRoughConductor(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalRoughConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfRoughConductor(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SamplePlastic(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;

    __device__ void SampleRoughPlastic(BsdfSampling &bs, const vec3 &sample) const;
    __device__ vec3 EvalRoughPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
    __device__ Float PdfRoughPlastic(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const;
};

__device__ Float Material::SpecularSamplingWeight(const vec2 &texcoord) const
{
    if (!diffuse_reflectance_ && !specular_reflectance_)
        return 2.0 / 3.0;
    else if (!diffuse_reflectance_)
    {
        auto ks = specular_reflectance_->Color(texcoord);
        auto s_sum = ks.x + ks.y + ks.z;
        return s_sum / (1.5 + s_sum);
    }
    else if (!specular_reflectance_)
    {
        auto kd = diffuse_reflectance_->Color(texcoord);
        auto d_sum = kd.x + kd.y + kd.z;
        return 3.0 / (d_sum + 3.0);
    }
    else
    {
        auto ks = specular_reflectance_->Color(texcoord);
        auto s_sum = ks.x + ks.y + ks.z;
        auto kd = diffuse_reflectance_->Color(texcoord);
        auto d_sum = kd.x + kd.y + kd.z;
        return s_sum / (d_sum + s_sum);
    }
}
__device__ Float Material::GetAlbedo(Float cos_theta) const
{
    auto offset = cos_theta * kAlbedoResolution;
    auto idx = static_cast<int>(offset);
    if (idx >= kAlbedoResolution - 1)
        return kulla_conty_table_[kAlbedoResolution - 1];
    else
        return (1 - (offset - idx)) * kulla_conty_table_[idx] +
               (offset - idx) * kulla_conty_table_[idx + 1];
}

__device__ vec3 Material::EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const
{
    auto albedo_i = GetAlbedo(abs(cos_i_n));
    auto albedo_o = GetAlbedo(abs(cos_o_n));
    auto f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
    return f_ms * f_add_;
}

__device__ vec3 Material::EvalMultipleScatter(Float cos_i_n, Float cos_o_n, int inside) const
{
    auto f_add = (inside == kTrue) ? f_add_inv_ : f_add_;
    auto albedo_i = GetAlbedo(abs(cos_i_n));
    auto albedo_o = GetAlbedo(abs(cos_o_n));
    auto f_ms = (1.0 - albedo_o) * (1.0 - albedo_i) / (kPi * (1.0 - albedo_avg_));
    return f_ms * f_add;
}