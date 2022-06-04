#pragma once

#include "bsdf_info.h"

class Bsdf
{
public:
    virtual __device__ ~Bsdf() {}

    /**
     * \brief 根据光线出射方向和表面法线方向，抽样光线入射方向
     * \param wo 光线出射方向
     * \param normal 表面法线方向
     * \param texcoord 表面纹理坐标，可选
     * \param inside 表面法线方向是否朝向表面内侧
     * \return 由 vec3 类型和 BsdfSamplingType 类型构成的 pair，分别代表抽样所得光线入射方向，和入射光线与出射光线之间的关系
     */
    virtual __device__ void Sample(SamplingRecord &rec, const vec3 &sample) const {}

    ///\brief 根据光线入射方向、出射方向和几何信息，计算光能衰减系数和相应的光线传播概率
    virtual __device__ void Eval(SamplingRecord &rec) const {}

    ///\return 辐射亮度
    virtual __device__ vec3 radiance() const { return vec3(0); }

    ///\return 是否发光
    __device__ bool HasEmission() const { return type_ == kAreaLight; }

    __device__ bool twosided() const { return twosided_; }

    __device__ bool HarshLobe() const { return type_ == kConductor; }

    __device__ bool BumpMapping() const { return bump_map_ != nullptr; }

    ///\return 是否映射纹理
    virtual __device__ bool TextureMapping() const { return opacity_map_ != nullptr || bump_map_ != nullptr; }

    virtual __device__ bool Transparent(const vec2 &texcoord, const vec2 &sample) const
    {
        if (opacity_map_)
        {
            if (opacity_map_->Transparent(texcoord, sample.x))
                return true;
        }
        return false;
    }

    __device__ vec3 PerturbNormal(const vec3 &normal, const vec3 &tangent, const vec3 &bitangent, const vec2 &texcoord) const
    {
        auto TBN = gmat3(gvec3(tangent.x, tangent.y, tangent.z),
                         gvec3(bitangent.x, bitangent.y, bitangent.z),
                         gvec3(normal.x, normal.y, normal.z));
        auto gradient = bump_map_->Gradient(texcoord);
        auto normal_pertubed_local = glm::normalize(gvec3(-gradient.x, -gradient.y, 1));
        auto normal_pertubed = glm::normalize(TBN * normal_pertubed_local);
        return vec3(normal_pertubed.x, normal_pertubed.y, normal_pertubed.z);
    }

protected:
    __device__ Bsdf(uint idx, BsdfType type, bool twosided, Texture *bump_map, Texture *opacity_map)
        : idx_(idx), type_(type), twosided_(twosided), bump_map_(bump_map), opacity_map_(opacity_map) {}

private:
    uint idx_;
    BsdfType type_;
    bool twosided_;
    Texture *bump_map_;
    Texture *opacity_map_;
};