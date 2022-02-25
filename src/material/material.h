#pragma once

#include <memory>
#include <utility>
#include <map>

#include "glm/gtx/matrix_query.hpp"
#include "../utils/global.h"
#include "texture/textures.h"
#include "../rendering/ray.h"

NAMESPACE_BEGIN(simple_renderer)

// 材质类型（表面散射模型类型）
enum class MaterialType
{
    kAreaLight,       //面光源
    kDiffuse,         //漫反射
    kGlossy,          //冯模型定义的有光泽表面，含有漫反射和镜面反射
    kDielectric,      //平滑的电介质
    kRoughDielectric, //粗糙的电介质
    kThinDielectric,  //薄的电介质
    kConductor,       //平滑的导体
    kRoughConductor,  //粗糙的导体
    kPlastic,         //平滑的塑料
    kRoughPlastic     //粗糙的塑料
};

//按 BSDF 采样记录
struct BsdfSampling
{
    Vector3 wi;      //光线入射方向
    Spectrum weight; // BSDF 系数
    Float pdf;       //光线从该方向入射的概率

    //按 BSDF 采样记录
    BsdfSampling() : wi(Vector3(0)), weight(Spectrum(0)), pdf(0) {}
};

class Material
{
public:
    ~Material()
    {
        if (opacity_map_)
            DeleteTexturePointer(opacity_map_);
    }

    /**
     * \brief 根据光线出射方向和表面法线方向，抽样光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 由 Vector3 类型和 BsdfSamplingType 类型构成的 pair，分别代表抽样所得光线入射方向，和入射光线与出射光线之间的关系
     */
    virtual BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const = 0;

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

    ///\return 材质类型
    MaterialType type() const { return type_; }

    ///\return 材质id
    std::string id() const { return id_; }

    ///\return 是否两面都有效
    bool twosided() const { return twosided_; }

    ///\brief 设置是否两面都有效
    void setTwosided(bool twosided) { twosided_ = twosided; }

    /**
     * \brief 设置不透明度
     * \param opacity 不透明度（注意：目前仅支持每个通道不透明度都相同的不透明度）
     */
    void setOpacity(const Vector3 &opacity) { opacity_ = std::make_unique<Float>(std::max(std::max(opacity.r, opacity.g), opacity.b)); }

    /**
     * \brief 设置不透明度
     * \param opacity_map 不透明度映射纹理（注意：目前仅支持每个通道不透明度都相同的不透明度映射纹理）
     */
    void setOpacity(Texture *opacity_map) { opacity_map_ = opacity_map; }

    ///\return 是否使用不透明度
    bool OpacityMapping() const
    {
        if (opacity_ || opacity_map_)
            return true;
        else
            return false;
    }

    /**
     * \param texcoord 纹理坐标
     * \return 材质在给定的纹理坐标处是否透明
     */
    virtual bool Transparent(const Vector2 &texcoord) const
    {
        if (opacity_)
        {
            auto x = UniformFloat();
            if (x < *opacity_)
                return false;
            else
                return true;
        }
        else if (opacity_map_)
        {
            auto x = UniformFloat();
            auto opacity = GetLuminance(opacity_map_->GetPixel(texcoord));
            if (x < opacity)
                return false;
            else
                return true;
        }
        else
            return false;
    }

    /**
     * \brief 设置凹凸映射
     * \param opacity_map 凹凸映射纹理
     */
    void setBump(Texture *bump_map) { bump_map_ = bump_map; }

    ///\return 是否通过纹理映射更改法线
    bool NormalPerturbing() const
    {
        if (bump_map_)
            return true;
        else
            return false;
    }

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
    Material(const std::string &id, MaterialType type)
        : id_(id),
          type_(type),
          twosided_(true),
          opacity_(nullptr),
          opacity_map_(nullptr),
          bump_map_(nullptr){};

private:
    std::string id_;                 // 材质id
    MaterialType type_;              // 材质类型（表面散射模型类型）
    bool twosided_;                  // 材质两面都有效
    std::unique_ptr<Float> opacity_; // 不透明度
    Texture *opacity_map_;           // 透明度纹理映射
    Texture *bump_map_;              // 透明度纹理映射
};

NAMESPACE_END(simple_renderer)