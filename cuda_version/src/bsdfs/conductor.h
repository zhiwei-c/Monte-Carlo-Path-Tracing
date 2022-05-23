#pragma once

#include "../core/material_base.h"

///\brief 平滑的导体材质派生类
class Conductor : public Material
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
        : Material(idx, kConductor, twosided, bump_map, opacity_map),
          mirror_(mirror), eta_(eta), k_(k), specular_reflectance_(specular_reflectance)
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const override
    {
        bs.pdf = 1;
        bs.wi = -Reflect(-bs.wo, bs.normal);
        bs.attenuation = vec3(1);
        if (specular_reflectance_)
            bs.attenuation = specular_reflectance_->Color(bs.texcoord);
        if (!mirror_)
            bs.attenuation *= FresnelConductor(bs.wi, bs.normal, eta_, k_);
        bs.valid = true;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        if (!SameDirection(wo, Reflect(wi, normal)))
            return vec3(0);
        auto albedo = vec3(1);
        if (specular_reflectance_)
            albedo = specular_reflectance_->Color(texcoord);
        if (!mirror_)
            albedo *= FresnelConductor(wi, normal, eta_, k_);
        return albedo;
    }

    ///\return 根据光线入射方向和法线方向计算的，光线从给定出射方向射出的概率
    __device__ Float Pdf(const vec3 &wi, const vec3 &wo, const vec3 &normal, const vec2 &texcoord, int inside) const override
    {
        if (SameDirection(wo, Reflect(wi, normal)))
            return 1;
        else
            return 0;
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Material::TextureMapping() ||
               specular_reflectance_ && specular_reflectance_->Varying();
    }

private:
    bool mirror_;                   //是否是镜面
    vec3 eta_;                      //复数折射率的实部
    vec3 k_;                        //复数折射率的虚部（消光系数）
    Texture *specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
};

__device__ inline void InitConductor(size_t m_idx,
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

    material_list[m_idx] = new Conductor(m_idx,
                                         material_info_list[m_idx].twosided,
                                         bump_map,
                                         opacity_map,
                                         material_info_list[m_idx].mirror,
                                         material_info_list[m_idx].eta,
                                         material_info_list[m_idx].k,
                                         specular_reflectance);
}