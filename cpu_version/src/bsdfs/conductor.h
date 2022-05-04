#pragma once

#include "../core/material_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 平滑的导体材质派生类
class Conductor : public Material
{
public:
    ///\brief 平滑的导体材质
    ///\param mirror 是否是镜面（全反射）
    ///\param eta 材质折射率的实部
    ///\param k 材质折射率的虚部（消光系数）
    ///\param ext_ior 外折射率
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    Conductor(bool mirror,
              const Spectrum &eta,
              const Spectrum &k,
              Float ext_ior,
              std::unique_ptr<Texture> specular_reflectance);

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override;

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 是否映射纹理
    bool TextureMapping() const override;

private:
    bool mirror_;                                   //是否是镜面
    Spectrum eta_;                                  //材质相对折射率的实部
    Spectrum k_;                                    //材质相对折射率的虚部（消光系数）
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
};

NAMESPACE_END(simple_renderer)