#pragma once

#include "../modeling/accelerator/aabb.h"
#include "../modeling/material.h"

class Intersection
{

public:
    ///\brief 光线与物体模型面片交点，光线与物体没有相交
    __device__ Intersection()
        : valid_(false),
          absorb_(false),
          pos_(vec3(0)),
          normal_(vec3(0)),
          inside_(0),
          distance_(INFINITY),
          texcoord_(vec2(0)),
          material_(nullptr),
          pdf_area_(INFINITY) {}

    __device__ Intersection(Float distance)
        : valid_(true),
          absorb_(true),
          pos_(vec3(0)),
          normal_(vec3(0)),
          inside_(0),
          distance_(distance),
          texcoord_(vec2(0)),
          material_(nullptr),
          pdf_area_(INFINITY) {}

    __device__ Intersection(const vec3 &pos,
                            const vec3 &normal,
                            const vec2 &texcoord,
                            int inside,
                            Float distance,
                            Material *material,
                            Float pdf_area)
        : valid_(true),
          absorb_(false),
          pos_(pos),
          normal_(normal),
          inside_(inside),
          distance_(distance),
          texcoord_(texcoord),
          material_(material),
          pdf_area_(pdf_area) {}

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
    __device__ bool HasEmission() const { return material_->HasEmission(); }

    ///\return 交点处的物体表面的辐射亮度
    __device__ vec3 radiance() const { return material_->radiance(); }

    __device__ bool HashLobe() const { return material_->HarshLobe(); }

    ///\brief 面元概率
    __device__ Float pdf_area() const { return pdf_area_; }

    /**
     * \brief 交点处给定光线出射方向，采样入射方向
     * \param wo 给定的光线出射方向
     * \return 采样得到的光线入射方向
     */
    __device__ void Sample(BsdfSampling &bs, const vec3 &sample) const
    {
        auto one_side = myvec::dot(bs.wo, normal_) > 0; //光线与交点法线是否同侧

        bs.inside = inside_;
        if (!one_side)
            bs.inside = (bs.inside == kTrue) ? kFalse: kTrue;

        bs.normal = one_side ? normal_ : -normal_;
        bs.texcoord = texcoord_;

        material_->Sample(bs, sample);

        if (bs.pdf < kEpsilonPdf)
            bs.valid = false;
    }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出，
     *      计算BSDF（bidirectional scattering distribution function，双向散射分别函数）系数
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的BSDF系数
     */
    __device__ vec3 Eval(const vec3 &wi, const vec3 &wo) const
    {
        auto one_side = myvec::dot(wi, normal_) < 0; //入射光线与法线是否同侧

        auto local_inside = inside_;
        if (!one_side)
            local_inside = (local_inside == kTrue) ? kFalse: kTrue;

        auto normal = one_side ? normal_ : -normal_;

        return material_->Eval(wi, wo, normal, texcoord_, local_inside);
    }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出的概率
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的概率
     */
    __device__ Float Pdf(const vec3 &wi, const vec3 &wo) const
    {
        auto one_side = myvec::dot(wi, normal_) < 0; //入射光线与法线是否同侧

        auto local_inside = inside_;
        if (!one_side)
            local_inside = (local_inside == kTrue) ? kFalse: kTrue;

        auto normal = one_side ? normal_ : -normal_;

        return material_->Pdf(wi, wo, normal, texcoord_, local_inside);
    }

private:
    bool valid_;         //光线与物体的相交是否发生
    bool absorb_;        //光线与单面材质的物体交于物体背面而被吸收
    int inside_;         //交点处法线是否朝内
    Float distance_;     //从光线起点到该交点的距离
    Float pdf_area_;     //面元概率
    vec2 texcoord_;      //交点纹理坐标
    vec3 pos_;           //交点空间坐标
    vec3 normal_;        //交点法线
    Material *material_; //交点面片对应的材质
};