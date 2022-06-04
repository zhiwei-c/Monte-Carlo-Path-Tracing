#pragma once

#include "../core/bsdf_base.h"

///\brief 平滑的电介质派生类
class Dielectric : public Bsdf
{
public:
    /**
     * @brief 平滑的电介质材质
     *
     * @param idx 材质编号
     * @param twosided 材质是否两面都有效
     * @param bump_map 不透明度纹理映射
     * @param opacity_map 凹凸映射
     * @param eta 相对折射率
     * @param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     * @param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
     */
    __device__ Dielectric(uint idx, bool twosided, Texture *bump_map, Texture *opacity_map,
                          vec3 eta, Texture *specular_reflectance, Texture *specular_transmittance)
        : Bsdf(idx, kDielectric, twosided, bump_map, opacity_map),
          eta_d_(eta.x), eta_inv_d_(1.0 / eta.x), specular_reflectance_(specular_reflectance),
          specular_transmittance_(specular_transmittance)
    {
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const override
    {
        Float eta = (rec.inside == kTrue) ? eta_inv_d_ : eta_d_,   //相对折射率，即光线透射侧介质折射率与入透射侧介质折射率之比
            eta_inv = (rec.inside == kTrue) ? eta_d_ : eta_inv_d_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(-rec.wo, rec.normal, eta_inv);            //菲涅尔项
        if (sample.x < kr)
        { //抽样反射光线
            rec.wi = -Reflect(-rec.wo, rec.normal);
            rec.pdf = kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.attenuation = vec3(kr);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else
        { //抽样折射光线
            rec.wi = -Refract(-rec.wo, rec.normal, eta_inv);
            { //光线折射时穿过了介质，为了使光线入射方向和表面法线方向夹角的余弦仍小于零，需做一些相应处理
                rec.normal = -rec.normal;
                rec.inside = static_cast<int>(!static_cast<bool>(rec.inside));
                eta_inv = eta;
            }
            kr = Fresnel(rec.wi, rec.normal, eta_inv);
            rec.pdf = 1.0 - kr;
            if (rec.pdf < kEpsilonPdf)
                return;
            rec.attenuation = vec3(1.0 - kr);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= eta_inv * eta_inv;
        }
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
        rec.valid = true;
    }

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    __device__ void Eval(SamplingRecord &rec) const override
    {
        Float eta_inv = (rec.inside == kTrue) ? eta_d_ : eta_inv_d_, //相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
            kr = Fresnel(rec.wi, rec.normal, eta_inv);
        if (SameDirection(rec.wo, Reflect(rec.wi, rec.normal)))
        {
            rec.pdf = kr;
            rec.attenuation = vec3(kr);
            if (specular_reflectance_)
                rec.attenuation *= specular_reflectance_->Color(rec.texcoord);
        }
        else if (SameDirection(rec.wo, Refract(rec.wi, rec.normal, eta_inv)))
        {
            rec.pdf = 1.0 - kr;
            rec.attenuation = vec3(1.0 - kr);
            if (specular_transmittance_)
                rec.attenuation *= specular_transmittance_->Color(rec.texcoord);
            //光线折射后，光路可能覆盖的立体角范围发生了改变，对辐射亮度进行积分需要进行相应的处理
            rec.attenuation *= eta_inv * eta_inv;
        }
        else
            return;
        rec.attenuation *= myvec::dot(-rec.wi, rec.normal);
        rec.valid = true;
    }

    ///\return 是否映射纹理
    __device__ bool TextureMapping() const override
    {
        return Bsdf::TextureMapping() || specular_reflectance_ && specular_reflectance_->Varying() ||
               specular_transmittance_ && specular_transmittance_->Varying();
    }

private:
    Float eta_d_;                     //介质折射率与外部折射率之比
    Float eta_inv_d_;                 //外部折射率与介质折射率之比
    Texture *specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
    Texture *specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针，不应更改此参数）
};

__device__ inline void InitDielectric(uint m_idx, BsdfInfo *bsdf_info_list, Texture *texture_list,
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

    bsdf_list[m_idx] = new Dielectric(m_idx, bsdf_info_list[m_idx].twosided, bump_map, opacity_map,
                                          bsdf_info_list[m_idx].eta, specular_reflectance, specular_transmittance);
}
