#pragma once

#include "../core/sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//材质类型
enum class BsdfType
{
    kDiffuse,         //平滑的理想漫反射材质，由朗伯模型描述
    kRoughDiffuse,    //粗糙的理想漫反射材质，由 Oren–Nayar Reflectance Model 描述
    kGlossy,          //有光泽的材质，由冯模型描述，含有漫反射和镜面反射
    kDielectric,      //平滑的电介质
    kRoughDielectric, //粗糙的电介质
    kThinDielectric,  //薄的电介质
    kConductor,       //平滑的导体
    kRoughConductor,  //粗糙的导体
    kClearCoatedConductor,  //粗糙的导体
    kPlastic,         //平滑的塑料
    kRoughPlastic,    //粗糙的塑料
};

//材质（表面散射模型）
class Bsdf
{
public:
    virtual ~Bsdf() {}

    virtual void Sample(SamplingRecord *rec, Sampler *sampler) const {};
    virtual void Eval(SamplingRecord *rec) const {};
    void ApplyBumpMapping(const dvec3 &tangent, const dvec3 &bitangent, const dvec2 &texcoord, dvec3 *normal) const;

    bool IsHarshLobe() const;
    bool IsEmitter() const { return is_emitter_; }
    bool IsTwosided() const { return twosided_; }
    virtual bool IsTransparent(const dvec2 &texcoord, Sampler *sampler) const;

    const std::string &id() const { return id_; }
    virtual dvec3 radiance() const { return radiance_; }

    void SetRadiance(const dvec3 &radiance);
    void SetTwosided(bool twosided) { twosided_ = twosided; }
    void SetOpacity(Texture *opacity) { opacity_ = opacity; }
    void SetBumpMapping(Texture *bump_map) { bump_map_ = bump_map; }

protected:
    Bsdf(BsdfType type, const std::string &id);

    virtual bool UseTextureMapping() const;

private:
    BsdfType type_;     // 材质类型（表面散射模型类型）
    bool is_emitter_;   // 是否发光
    bool twosided_;     // 材质两面都有效
    Texture *opacity_;  // 不透明度纹理映射
    Texture *bump_map_; // 凹凸映射
    dvec3 radiance_;    // 作为光源时的出射辐射亮度
    std::string id_;    // 材质ID
};

NAMESPACE_END(raytracer)