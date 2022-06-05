#pragma once

#include <memory>
#include <utility>
#include <map>

#include "glm/gtx/matrix_query.hpp"

#include "texture.h"
#include "ray.h"
#include "sampling_record.h"

NAMESPACE_BEGIN(raytracer)

// 材质类型（表面散射模型类型）
enum class BsdfType
{
    kAreaLight,       //面光源
    kDiffuse,         //平滑的理想漫反射的表面，由朗伯模型描述
    kRoughDiffuse,    //粗糙的理想漫反射的表面，由 Oren–Nayar Reflectance Model 描述
    kGlossy,          //有光泽的表面，由冯模型描述，含有漫反射和镜面反射
    kDielectric,      //平滑的电介质
    kRoughDielectric, //粗糙的电介质
    kThinDielectric,  //薄的电介质
    kConductor,       //平滑的导体
    kRoughConductor,  //粗糙的导体
    kPlastic,         //平滑的塑料
    kRoughPlastic,    //粗糙的塑料
};


//材质基类
class Bsdf
{
public:
    virtual ~Bsdf() {}

    ///\brief 根据光线出射方向和几何信息，抽样光线入射方向
    virtual void Sample(SamplingRecord &rec) const {};

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    virtual void Eval(SamplingRecord &rec) const {}

    ///\return 材质对应的散射波瓣是是否是 δ-函数
    bool HarshLobe() const { return type_ == BsdfType::kConductor; }

    ///\return 是否发光
    virtual bool HasEmission() const { return type_ == BsdfType::kAreaLight; }

    ///\return 辐射亮度
    virtual Spectrum radiance() const { return Spectrum(0); };

    ///\return 是否映射纹理
    virtual bool TextureMapping() const { return opacity_ != nullptr || bump_map_ != nullptr; }

    ///\brief 设置是否两面都有效
    void SetTwosided(bool twosided) { twosided_ = twosided; }

    ///\return 是否两面都有效
    bool Twosided() const { return twosided_; }

    ///\brief 设置不透明度
    void SetOpacity(std::unique_ptr<Texture> opacity) { opacity_.reset(opacity.release()); }

    ///\return 材质在给定的纹理坐标处是否透明
    virtual bool Transparent(const Vector2 &texcoord) const { return opacity_ && UniformFloat() > opacity_->Color(texcoord).x; }

    ///\brief 设置凹凸映射
    void SetBumpMapping(std::unique_ptr<Texture> bump_map) { bump_map_.reset(bump_map.release()); }

    ///\return 是否通过纹理映射更改法线
    bool NormalPerturbing() const { return bump_map_ != nullptr; }

    ///\brief 通过凹凸映射更改法线
    ///\param normal 初始法线
    ///\param tangent 切线
    ///\param bitangent 副切线
    ///\param texcoord 纹理坐标
    ///\return 映射后的法线
    Vector3 PerturbNormal(const Vector3 &normal, const Vector3 &tangent, const Vector3 &bitangent, const Vector2 &texcoord) const
    {
        if (bump_map_ == nullptr)
            return normal;
        auto TBN = Mat3(tangent, bitangent, normal);
        Vector2 gradient = bump_map_->Gradient(texcoord);
        Vector3 normal_pertubed_local = glm::normalize(Vector3(-gradient.x, -gradient.y, 1));
        return glm::normalize(TBN * normal_pertubed_local);
    }

protected:
    ///\brief 材质基类
    ///\param type 材质类型
    Bsdf(BsdfType type) : type_(type), twosided_(false), opacity_(nullptr), bump_map_(nullptr){};

private:
    BsdfType type_;                     // 材质类型（表面散射模型类型）
    bool twosided_;                     // 材质两面都有效
    std::unique_ptr<Texture> opacity_;  // 不透明度纹理映射
    std::unique_ptr<Texture> bump_map_; // 凹凸映射
};

NAMESPACE_END(raytracer)