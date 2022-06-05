#pragma once

#include "../accelerator/aabb.h"
#include "bsdf.h"

class Intersection
{

public:
    ///\brief 光线与物体模型面片交点，光线与物体没有相交
    __device__ Intersection()
        : valid_(false), absorb_(false), pos_(vec3(0)), normal_(vec3(0)), inside_(false),
          distance_(INFINITY), texcoord_(vec2(0)), bsdf_(nullptr), pdf_area_(INFINITY)
    {
    }

    __device__ Intersection(Float distance)
        : valid_(true), absorb_(true), pos_(vec3(0)), normal_(vec3(0)), inside_(false),
          distance_(distance), texcoord_(vec2(0)), bsdf_(nullptr), pdf_area_(INFINITY)
    {
    }

    ///\brief 视点或散射点
    ///\param pos 视点或散射点的位置
    ///\param medium 视点或散射点的所处的介质
    __device__ Intersection(const vec3 &pos)
        : valid_(true), absorb_(false), pos_(pos), normal_(vec3(0)), inside_(false),
          distance_(INFINITY), texcoord_(vec2(0)), bsdf_(nullptr), pdf_area_(INFINITY)
    {
    }

    __device__ Intersection(const vec3 &pos, const vec3 &normal, const vec2 &texcoord, bool inside,
                            Float distance, Bsdf **bsdf, Float pdf_area)
        : valid_(true), absorb_(false), pos_(pos), normal_(normal), inside_(inside),
          distance_(distance), texcoord_(texcoord), bsdf_(bsdf), pdf_area_(pdf_area)
    {
    }

    ///\return 交点的位置
    __device__ vec3 pos() const { return pos_; }

    ///\return 交点处的物体表面的法线
    __device__ vec3 normal() const { return normal_; }

    ///\return 光线与物体的相交是否发生
    __device__ bool valid() const { return valid_; }

    __device__ bool absorb() const { return absorb_; }

    ///\return 从光线起点到该交点的距离
    __device__ Float distance() const { return distance_; }

    ///\return 交点处的物体表面是否发光
    __device__ bool HasEmission() const { return (*bsdf_)->HasEmission(); }

    ///\return 交点处的物体表面的辐射亮度
    __device__ vec3 radiance() const { return (*bsdf_)->radiance(); }

    __device__ bool HashLobe() const { return bsdf_ && (*bsdf_)->HarshLobe(); }

    ///\brief 面元概率
    __device__ Float pdf_area() const { return pdf_area_; }

    /**
     * \brief 交点处给定光线出射方向，采样入射方向
     * \param wo 给定的光线出射方向
     * \return 采样得到的光线入射方向
     */
    __device__ SamplingRecord Sample(const vec3 &wo, const vec3 &sample) const
    {
        auto one_side = myvec::dot(wo, normal_) > 0; //光线与交点法线是否同侧
        auto rec = SamplingRecord();
        rec.pos = pos_;
        rec.wo = wo;
        if (bsdf_)
        {
            rec.inside = one_side ? inside_ : !inside_;
            rec.normal = one_side ? normal_ : -normal_;
            rec.texcoord = texcoord_;
            (*bsdf_)->Sample(rec, sample);
        }
        else
        {
            rec.wi = wo;
            rec.pdf = 1;
            rec.attenuation = vec3(1);
            rec.valid = true;
        }
        return rec;
    }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出，
     *      计算BSDF（bidirectional scattering distribution function，双向散射分别函数）系数
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的BSDF系数
     */
    __device__ SamplingRecord Eval(const vec3 &wi, const vec3 &wo) const
    {
        auto rec = SamplingRecord();
        rec.pos = pos_;
        rec.wo = wo;
        rec.wi = wi;
        if (bsdf_)
        {
            auto one_side = myvec::dot(wi, normal_) < 0; //入射光线与法线是否同侧
            rec.inside = one_side ? inside_ : !inside_;
            rec.normal = one_side ? normal_ : -normal_;
            rec.texcoord = texcoord_;
            (*bsdf_)->Eval(rec);
        }
        else
        {
            rec.pdf = 1;
            rec.attenuation = vec3(1);
            rec.valid = true;
        }
        return rec;
    }

    bool Inner(const vec3 &wo) const
    {
        if (myvec::dot(wo, normal_) > 0)
            return inside_;
        else
            return !inside_;
    }

private:
    bool valid_;     //光线与物体的相交是否发生
    bool absorb_;    //光线与单面材质的物体交于物体背面而被吸收
    bool inside_;    //交点处法线是否朝内
    Float distance_; //从光线起点到该交点的距离
    Float pdf_area_; //面元概率
    vec2 texcoord_;  //交点纹理坐标
    vec3 pos_;       //交点空间坐标
    vec3 normal_;    //交点法线
    Bsdf **bsdf_;    //交点面片对应的材质
};