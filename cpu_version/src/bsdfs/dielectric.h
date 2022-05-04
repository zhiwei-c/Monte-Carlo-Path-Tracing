#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的电介质派生类
class Dielectric : public Material
{
public:
    ///\brief 平滑的电介质材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Dielectric(Float int_ior,
               Float ext_ior,
               std::unique_ptr<Texture> specular_reflectance,
               std::unique_ptr<Texture> specular_transmittance);

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override;

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 是否映射纹理
    bool TextureMapping() const override;

private:
    Float eta_;                                       //光线射入材质的相对折射率
    Float eta_inv_;                                   //光线从材质内部射出的相对折射率
    std::unique_ptr<Texture> specular_reflectance_;   //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    std::unique_ptr<Texture> specular_transmittance_; //镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(simple_renderer)