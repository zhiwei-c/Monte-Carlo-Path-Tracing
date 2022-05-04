#pragma once

#include "../core/material_base.h"
#include "../core/microfacet.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 粗糙的电介质派生类
class RoughDielectric : public Material, public Microfacet
{
public:
	///\brief 粗糙的电介质材质
	///\param distrib_type 用于模拟表面粗糙度的微表面分布的类型
	///\param alpha_u 沿切线（tangent）方向的粗糙度
	///\param alpha_v 沿副切线（bitangent）方向的粗糙度
	///\param int_ior 内折射率
	///\param ext_ior 外折射率
	///\param specular_reflectance 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	///\param specular_transmittance 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	RoughDielectric(Float int_ior,
					Float ext_ior,
					std::unique_ptr<Texture> specular_reflectance,
					std::unique_ptr<Texture> specular_transmittance,
					MicrofacetDistribType distrib_type,
					std::unique_ptr<Texture> alpha_u,
					std::unique_ptr<Texture> alpha_v);

	///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
	void Sample(BsdfSampling &bs) const override;

	///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
	Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

	///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
	Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const override;

	///\brief 是否映射纹理
	bool TextureMapping() const override;

private:
	Float eta_;										  //光线射入材质的相对折射率
	Float eta_inv_;									  //光线射出材质的相对折射率
	Float f_add_;									  //入射光线在物体外部，补偿多次散射后出射光能的系数
	Float f_add_inv_;								  //入射光线在物体内部，补偿多次散射后出射光能的系数
	Float ratio_t_;									  //入射光线在物体外部，补偿多次散射后出射光能中折射的比例
	Float ratio_t_inv_;								  //入射光线在物体内部，补偿多次散射后出射光能中折射的比例
	std::unique_ptr<Texture> specular_reflectance_;	  // 镜面反射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）
	std::unique_ptr<Texture> specular_transmittance_; // 镜面透射系数 （注意：对于物理真实感绘制，默认为 1，表示为空指针）

	///\brief 补偿多次散射后又射出的光能
	Float EvalMultipleScatter(Float cos_i_n, Float cos_o_n, bool inside) const;
};

NAMESPACE_END(simple_renderer)