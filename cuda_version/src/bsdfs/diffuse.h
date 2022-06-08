#pragma once

#include "../core/bsdf.h"

///\brief 平滑的理想漫反射材质派生类
class Diffuse : public Bsdf
{
public:
    /**
     * @brief 平滑的理想漫反射材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param diffuse_reflectance 漫反射系数
     */
    __device__ Diffuse(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                       Texture *diffuse_reflectance)
        : Bsdf(idx, kDiffuse, twosided, bump_map, opacity_map),
          diffuse_reflectance_(diffuse_reflectance) {}

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const override
    {
        auto wi_local = vec3(0);
        Float pdf = 0;
        HemisCos(sample.x, sample.y, wi_local, pdf);
        if (pdf < kEpsilonPdf)
            return;
        rec.wi = -ToWorld(wi_local, rec.normal);
        rec.pdf = pdf;
        rec.valid = true;

        if (diffuse_reflectance_ != nullptr)
            rec.attenuation = diffuse_reflectance_->Color(rec.texcoord) * kPiInv * myvec::dot(-rec.wi, rec.normal);
        else
            rec.attenuation = vec3(0.5 * kPiInv) * myvec::dot(-rec.wi, rec.normal);
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        // 表面法线方向，光线入射和出射需在介质同侧
        if (NotSameHemis(rec.wo, rec.normal))
            return;
        rec.pdf = PdfHemisCos(ToLocal(rec.wo, rec.normal));
        if (rec.pdf < kEpsilonPdf)
            return;
        rec.valid = true;

        if (diffuse_reflectance_ != nullptr)
            rec.attenuation = diffuse_reflectance_->Color(rec.texcoord) * kPiInv * myvec::dot(-rec.wi, rec.normal);
        else
            rec.attenuation = vec3(0.5 * kPiInv) * myvec::dot(-rec.wi, rec.normal);
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

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || diffuse_reflectance_ && diffuse_reflectance_->Varying();
    }

private:
    Texture *diffuse_reflectance_; //漫反射系数
};

__device__ inline void InitDiffuse(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
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

    bsdf_list[m_idx] = new Diffuse(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map, diffuse_reflectance);
}