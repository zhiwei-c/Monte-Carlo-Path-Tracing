#pragma once

#include "material_base.h"
#include "../accelerator/aabb.h"

NAMESPACE_BEGIN(raytracer)

//光线与面片交点类
class Intersection
{
public:
    ///\brief 光线与物体模型面片交点，光线与物体没有相交
    Intersection()
        : valid_(false), absorb_(false), pos_(Vector3(0)), normal_(Vector3(0)), inside_(false),
          distance_(INFINITY), texcoord_(Vector2(0)), material_(nullptr), pdf_area_(INFINITY)
    {
    }

    ///\brief 光线与物体模型面片交点，光线与材质是单面的物体相交于背面
    ///\param distance 光线起点与交点间的距离
    Intersection(Float distance)
        : valid_(true), absorb_(true), pos_(Vector3(0)), normal_(Vector3(0)), inside_(false),
          distance_(distance), texcoord_(Vector2(0)), material_(nullptr), pdf_area_(INFINITY)
    {
    }

    ///\brief 光线与物体模型面片交点，光线与物体相交
    ///\param pos 空间坐标
    ///\param normal 法线
    ///\param texcoord 纹理坐标
    ///\param inside 法线是否朝内
    ///\param distance 光线起点与交点间的距离
    ///\param material 交点处物体表面的材质
    ///\param pdf_area 交点处面元对应的概率
    Intersection(const Vector3 &pos, const Vector3 &normal, const Vector2 &texcoord, bool inside,
                 Float distance, Material *material, Float pdf_area)
        : valid_(true), absorb_(false), pos_(pos), normal_(normal), inside_(inside),
          distance_(distance), texcoord_(texcoord), material_(material), pdf_area_(pdf_area)
    {
    }

    ///\brief 给定交点处光线出射方向，抽样入射方向和相应的概率等参数
    std::unique_ptr<BsdfSampling> Sample(const Vector3 &wo, bool get_attenuation = true) const
    {
        bool one_side = glm::dot(wo, normal_) > 0; //光线与交点法线是否同侧
        auto bs = std::make_unique<BsdfSampling>();
        bs->inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
        bs->get_attenuation = get_attenuation;
        bs->texcoord = texcoord_;
        bs->wo = wo;
        bs->normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线出射方向夹角小于90度
        material_->Sample(*bs);
        if (bs->pdf < kEpsilonPdf)
            return nullptr;
        else
            return bs;
    }

    ///\brief 根据光线入射方向和出射方向，计算交点处光线传播的 BSDF 系数
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo) const
    {
        bool one_side = glm::dot(wi, normal_) < 0;      //入射光线与法线是否同侧
        bool inside = one_side ? inside_ : !inside_;    //法线方向是否指向介质内侧
        Vector3 normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线入射方向夹角大于90度
        return material_->Eval(wi, wo, normal, texcoord_, inside);
    }

    ///\brief 根据光线入射方向和出射方向，计算交点处光线传播的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo) const
    {
        bool one_side = glm::dot(wi, normal_) < 0;      //入射光线与法线是否同侧
        bool inside = one_side ? inside_ : !inside_;    //法线方向是否指向介质内侧
        Vector3 normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线入射方向夹角大于90度
        return material_->Pdf(wi, wo, normal, texcoord_, inside);
    }

    ///\return 交点的位置
    Vector3 pos() const { return pos_; }

    ///\return 交点处的物体表面的法线
    Vector3 normal() const { return normal_; }

    ///\return 光线与物体的相交是否发生
    bool valid() const { return valid_; }

    ///\return 光线与单面材质的物体交于物体背面而被吸收
    bool absorb() const { return absorb_; }

    ///\return 从光线起点到该交点的距离
    Float distance() const { return distance_; }

    ///\return 交点处的物体表面是否发光
    bool HasEmission() const { return material_->HasEmission(); }

    ///\return 交点处的物体表面的辐射亮度
    Spectrum radiance() const { return material_->radiance(); }

    ///\brief 面元概率
    Float pdf_area() const { return pdf_area_; }

    ///\return 交点处表面材质对应的散射波瓣分布是否是 δ-函数
    bool HarshLobe() const { return material_->HarshLobe(); }

private:
    bool valid_;         //光线与物体的相交是否发生
    bool inside_;        //交点处法线是否朝内
    bool absorb_;        //光线与单面材质的物体交于物体背面而被吸收
    Float distance_;     //从光线起点到该交点的距离
    Float pdf_area_;     //面元概率
    Vector2 texcoord_;   //交点纹理坐标
    Vector3 pos_;        //交点空间坐标
    Vector3 normal_;     //交点法线
    Material *material_; //交点面片对应的材质
};

NAMESPACE_END(raytracer)