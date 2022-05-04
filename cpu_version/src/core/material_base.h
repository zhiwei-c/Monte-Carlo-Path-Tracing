#pragma once

#include <memory>
#include <utility>
#include <map>

#include "glm/gtx/matrix_query.hpp"

#include "texture.h"
#include "ray.h"

NAMESPACE_BEGIN(simple_renderer)

// 材质类型（表面散射模型类型）
enum class MaterialType
{
    kAreaLight,       //面光源
    kDiffuse,         //漫反射
    kGlossy,          //冯模型描述的有光泽表面，含有漫反射和镜面反射
    kDielectric,      //平滑的电介质
    kRoughDielectric, //粗糙的电介质
    kThinDielectric,  //薄的电介质
    kConductor,       //平滑的导体
    kRoughConductor,  //粗糙的导体
    kPlastic,         //平滑的塑料
    kRoughPlastic,    //粗糙的塑料
};

//按 BSDF 采样记录
struct BsdfSampling
{
    bool inside;          //表面法线方向是否朝向表面内侧
    bool get_attenuation; //是否计算 BSDF 系数
    Float pdf;            //光线从该方向入射的概率
    Vector2 texcoord;     //表面纹理坐标
    Vector3 wi;           //光线入射方向
    Vector3 wo;           //光线出射方向
    Vector3 normal;       //表面法线方向
    Spectrum attenuation; // BSDF 系数

    BsdfSampling() : inside(false), get_attenuation(true), pdf(0), texcoord(Vector2(0)), wi(Vector3(0)), wo(Vector3(0)), normal(Vector3(0)), attenuation(Spectrum(0)) {}
};

//材质基类
class Material
{
public:
    virtual ~Material() {}

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    ///\param wo 光线出射方向
    ///\param normal 表面法线方向
    ///\param texcoord 表面纹理坐标，可选
    ///\param inside 表面法线方向是否朝向表面内侧
    ///\return 由 Vector3 类型和 BsdfSamplingType 类型构成的 pair，分别代表抽样所得光线入射方向，和入射光线与出射光线之间的关系
    virtual void Sample(BsdfSampling &bs) const {};

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    ///\param wi 光线入射方向
    ///\param wo 光线出射方向
    ///\param normal 表面法线方向
    ///\param texcoord 表面纹理坐标，可选
    ///\param inside 表面法线方向是否朝向表面内侧
    ///\return BSDF 权重
    virtual Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const { return Spectrum(0); }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算光线因从入射方向入射，而从出射方向出射的概率
    ///\param wi 光线入射方向
    ///\param wo 光线出射方向
    ///\param normal 表面法线方向
    ///\param texcoord 表面纹理坐标，可选
    ///\param inside 表面法线方向是否朝向表面内侧
    ///\return 光线因从入射方向入射，而从出射方向出射的概率
    virtual Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 &texcoord, bool inside) const { return 0; }

    ///\return 材质对应的散射波瓣是是否是 δ-函数
    bool HarshLobe() const { return type_ == MaterialType::kConductor; }

    ///\return 是否发光
    virtual bool HasEmission() const { return type_ == MaterialType::kAreaLight; }

    ///\return 辐射亮度
    virtual Spectrum radiance() const { return Spectrum(0); };

    ///\return 是否映射纹理
    virtual bool TextureMapping() const { return opacity_ && !opacity_->Constant() || bump_map_ && !bump_map_->Constant(); }

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
        auto gradient = bump_map_->Gradient(texcoord);
        auto normal_pertubed_local = glm::normalize(Vector3(-gradient.x, -gradient.y, 1));
        auto normal_pertubed = glm::normalize(TBN * normal_pertubed_local);
        return normal_pertubed;
    }

protected:
    ///\brief 材质基类
    ///\param type 材质类型
    Material(MaterialType type)
        : type_(type),
          twosided_(false),
          opacity_(nullptr),
          bump_map_(nullptr){};

private:
    MaterialType type_;                 // 材质类型（表面散射模型类型）
    bool twosided_;                     // 材质两面都有效
    std::unique_ptr<Texture> opacity_;  // 不透明度纹理映射
    std::unique_ptr<Texture> bump_map_; // 透明度纹理映射
};

NAMESPACE_END(simple_renderer)