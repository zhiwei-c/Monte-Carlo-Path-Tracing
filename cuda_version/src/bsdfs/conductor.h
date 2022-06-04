#pragma once

#include "../core/bsdf_base.h"

///\brief 平滑的导体材质派生类
class Conductor : public Bsdf
{
public:
    /**
     * @brief 平滑的导体材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param mirror 是否是镜面
     * @param eta 复数折射率的实部
     * @param k 复数折射率的虚部（消光系数）
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     */
    __device__ Conductor(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                         bool mirror, vec3 eta, vec3 k, Texture *specular_reflectance)
        : Bsdf(idx, kConductor, twosided, bump_map, opacity_map),
          mirror_(mirror), eta_(eta), k_(k), specular_reflectance_(specular_reflectance)
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const override
    {
        rec.pdf = 1;
        rec.wi = -Reflect(-rec.wo, rec.normal);
        rec.valid = true;
        rec.attenuation = vec3(1);
        if (specular_reflectance_)
            rec.attenuation = specular_reflectance_->Color(rec.texcoord);
        if (!mirror_)
            rec.attenuation *= FresnelConductor(rec.wi, rec.normal, eta_, k_);
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        if (!SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
            return;
        rec.pdf = 1;
        rec.valid = true;
        
        rec.attenuation = vec3(1);
        if (specular_reflectance_)
            rec.attenuation = specular_reflectance_->Color(rec.texcoord);
        if (!mirror_)
            rec.attenuation *= FresnelConductor(rec.wi, rec.normal, eta_, k_);
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
    }
    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && specular_reflectance_->Varying();
    }

private:
    bool mirror_;                   //是否是镜面
    vec3 eta_;                      //复数折射率的实部
    vec3 k_;                        //复数折射率的虚部（消光系数）
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
};

__device__ inline void InitConductor(size_t m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
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

    bsdf_list[m_idx] = new Conductor(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                         bsdf_info_list[m_idx].mirror, bsdf_info_list[m_idx].eta,
                                         bsdf_info_list[m_idx].k, specular_reflectance);
}