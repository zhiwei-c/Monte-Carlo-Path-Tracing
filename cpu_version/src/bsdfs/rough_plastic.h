#pragma once

#include "../core/material_base.h"
#include "../core/microfacet.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 粗糙的塑料材质派生类
class RoughPlastic : public Material, public Microfacet
{
public:
    ///\brief 粗糙的塑料材质
    ///\param int_ior 内折射率
    ///\param ext_ior 外折射率
    ///\param diffuse_reflectance 漫反射分量
    ///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
    ///\param distrib_type 用于模拟表面粗糙度的微表面分布的类型
    ///\param alpha 材质的粗糙度
    ///\param nonlinear 是否考虑因内部散射而引起的非线性色移
    RoughPlastic(Float int_ior,
                 Float ext_ior,
                 std::unique_ptr<Texture> diffuse_reflectance,
                 std::unique_ptr<Texture> specular_reflectance,
                 MicrofacetDistribType distrib_type,
                 std::unique_ptr<Texture> alpha,
                 bool nonlinear);

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    void Sample(BsdfSampling &bs) const override;

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

    ///\brief 是否映射纹理
    bool TextureMapping() const override;

    ///\brief 给定点是否透明
    bool Transparent(const Vector2 &texcoord) const override;

private:
    bool nonlinear_;                                //是否考虑因内部散射而引起的非线性色移
    Float eta_inv_;                                 //外部折射率与介质折射率之比
    Float fdr_;                                     //漫反射菲涅尔项平均值
    Float specular_sampling_weight_;                //抽样镜面反射权重
    Float f_add_;                                   //补偿多次散射后出射光能的系数
    std::unique_ptr<Texture> diffuse_reflectance_;  //漫反射系数
    std::unique_ptr<Texture> specular_reflectance_; //镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）

    ///\brief 补偿多次散射后又射出的光能
    Float EvalMultipleScatter(Float cos_i_n, Float cos_o_n) const;

    ///\brief 获取给定点抽样镜面反射的权重
    Float SpecularSamplingWeight(const Vector2 &texcoord) const;
};

NAMESPACE_END(simple_renderer)