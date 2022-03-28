#pragma once

#include <memory>
#include <utility>
#include <map>

#include "glm/gtx/matrix_query.hpp"
#include "../utils/global.h"
#include "textures/textures.h"
#include "../rendering/ray.h"

NAMESPACE_BEGIN(simple_renderer)

// 材质类型（表面散射模型类型）
enum class MaterialType
{
    kDiffuse,         //漫反射
    kGlossy,          //冯模型定义的有光泽表面，含有漫反射和镜面反射
    kDielectric,      //平滑的电介质
    kRoughDielectric, //粗糙的电介质
    kThinDielectric,  //薄的电介质
    kConductor,       //平滑的导体
    kRoughConductor,  //粗糙的导体
    kPlastic,         //平滑的塑料
    kRoughPlastic,    //粗糙的塑料

    kAreaLight //面光源
};

//按 BSDF 采样记录
struct BsdfSampling
{
    bool inside;             //表面法线方向是否朝向表面内侧
    bool get_weight;         //是否计算 BSDF 系数
    Float pdf;               //光线从该方向入射的概率
    const Vector2 *texcoord; //表面纹理坐标，可选
    Vector3 wi;              //光线入射方向
    Vector3 wo;              //光线出射方向
    Vector3 normal;          //表面法线方向
    Spectrum weight;         // BSDF 系数

    BsdfSampling() : inside(false), get_weight(true), pdf(0), texcoord(nullptr), wi(Vector3(0)), wo(Vector3(0)), normal(Vector3(0)), weight(Spectrum(0)) {}
};

class Material
{
public:
    virtual ~Material() {}

    /**
     * \brief 根据光线出射方向和表面法线方向，抽样光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 由 Vector3 类型和 BsdfSamplingType 类型构成的 pair，分别代表抽样所得光线入射方向，和入射光线与出射光线之间的关系
     */
    virtual void Sample(BsdfSampling &bs) const {};

    /**
     * \brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
     * \param wi 光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return BSDF 权重
     */
    virtual Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const = 0;

    /**
     * \brief 根据光线入射方向、出射方向和法线方向，计算光线因从入射方向入射，而从出射方向出射的概率
     * \param wi 光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 光线因从入射方向入射，而从出射方向出射的概率
     */
    virtual Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const = 0;

    ///\return 辐射亮度
    virtual Spectrum radiance() const { return Spectrum(0); };

    ///\return 是否发光
    virtual bool HasEmission() const { return false; }

    ///\return 是否映射纹理
    virtual bool TextureMapping() const { return false; }

    ///\return 是否两面都有效
    bool twosided() const { return twosided_; }

    ///\brief 设置是否两面都有效
    void setTwosided(bool twosided) { twosided_ = twosided; }

    /**
     * \brief 设置不透明度
     * \param opacity 不透明度（注意：目前仅支持每个通道不透明度都相同的不透明度）
     */
    void setOpacity(const Vector3 &opacity)
    {
        opacity_.reset(new ConstantTexture(opacity));
    }

    /**
     * \brief 设置不透明度
     * \param opacity_map 不透明度映射纹理（注意：目前仅支持每个通道不透明度都相同的不透明度映射纹理）
     */
    void setOpacity(std::unique_ptr<Texture> opacity)
    {
        opacity_.reset(opacity.release());
    }

    ///\return 是否使用不透明度
    bool OpacityMapping() const { return opacity_ != nullptr; }

    /**
     * \param texcoord 纹理坐标
     * \return 材质在给定的纹理坐标处是否透明
     */
    virtual bool Transparent(const Vector2 &texcoord) const
    {
        return opacity_ && UniformFloat() > GetLuminance(opacity_->GetPixel(texcoord));
    }

    /**
     * \brief 设置凹凸映射
     * \param opacity_map 凹凸映射纹理
     */
    void setBump(std::unique_ptr<Texture> bump_map) { bump_map_.reset(bump_map.release()); }

    ///\return 是否通过纹理映射更改法线
    bool NormalPerturbing() const { return bump_map_ != nullptr; }

    /**
     * \brief 通过纹理映射更改法线
     * \param normal 初始法线
     * \param tangent 切线
     * \param bitangent 副切线
     * \param texcoord 纹理坐标
     * \return 映射后的法线
     */
    Vector3 PerturbNormal(const Vector3 &normal, const Vector3 &tangent, const Vector3 &bitangent, const Vector2 &texcoord) const
    {
        if (bump_map_ == nullptr)
            return normal;
        auto TBN = Mat3(tangent, bitangent, normal);
        auto gradient = bump_map_->GetGradient(texcoord);
        auto normal_pertubed_local = glm::normalize(Vector3(-gradient.x, -gradient.y, 1));
        auto normal_pertubed = glm::normalize(TBN * normal_pertubed_local);
        return normal_pertubed;
    }

protected:
    Material(MaterialType type)
        : type_(type),
          twosided_(true),
          opacity_(nullptr),
          bump_map_(nullptr){};

private:
    bool twosided_;                     // 材质两面都有效
    MaterialType type_;                 // 材质类型（表面散射模型类型）
    std::string id_;                    // 材质id
    std::unique_ptr<Texture> opacity_;  // 不透明度
    std::unique_ptr<Texture> bump_map_; // 透明度纹理映射
};

NAMESPACE_END(simple_renderer)