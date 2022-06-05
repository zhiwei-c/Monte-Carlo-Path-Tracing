#pragma once

#include "bsdf_base.h"
#include "medium_base.h"
#include "../accelerator/aabb.h"

NAMESPACE_BEGIN(raytracer)

enum class IntersectionType
{
    kNone,
    kViewPoint,
    kMedium,
    kSurface,
    kAbsorbSurface,
};

//光线与面片交点类
class Intersection
{
public:
    ///\brief 光线与物体模型面片交点，光线与物体没有相交
    Intersection()
        : type_(IntersectionType::kNone), pos_(Vector3(0)), normal_(Vector3(0)), inside_(false), distance_(INFINITY),
          texcoord_(Vector2(0)), bsdf_(nullptr), medium_(nullptr), int_medium_(nullptr), ext_medium_(nullptr),
          pdf_area_(INFINITY)
    {
    }

    ///\brief 光线与物体模型面片交点，光线与材质是单面的物体相交于背面
    ///\param distance 光线起点与交点间的距离
    Intersection(Float distance)
        : type_(IntersectionType::kAbsorbSurface), pos_(Vector3(0)), normal_(Vector3(0)), inside_(false),
          distance_(distance), texcoord_(Vector2(0)), bsdf_(nullptr), medium_(nullptr), int_medium_(nullptr),
          ext_medium_(nullptr), pdf_area_(INFINITY)
    {
    }

    ///\brief 视点
    ///\param pos 视点的位置
    Intersection(const Vector3 &pos)
        : type_(IntersectionType::kViewPoint), pos_(pos), normal_(Vector3(0)), inside_(false), distance_(INFINITY),
          texcoord_(Vector2(0)), bsdf_(nullptr), medium_(nullptr), int_medium_(nullptr), ext_medium_(nullptr),
          pdf_area_(INFINITY)
    {
    }

    ///\brief 散射点
    ///\param pos 散射点的位置
    ///\param medium 散射点的所处的介质
    Intersection(const Vector3 &pos, Medium *medium)
        : type_(IntersectionType::kMedium), pos_(pos), normal_(Vector3(0)), inside_(false), distance_(INFINITY),
          texcoord_(Vector2(0)), bsdf_(nullptr), medium_(medium), int_medium_(nullptr), ext_medium_(nullptr),
          pdf_area_(INFINITY)
    {
    }

    ///\brief 光线与物体模型面片交点，光线与物体相交
    ///\param pos 空间坐标
    ///\param normal 法线
    ///\param texcoord 纹理坐标
    ///\param inside 法线是否朝内
    ///\param distance 光线起点与交点间的距离
    ///\param bsdf 交点处物体表面的材质
    ///\param medium 物体内部的介质
    ///\param pdf_area 交点处面元对应的概率
    Intersection(const Vector3 &pos, const Vector3 &normal, const Vector2 &texcoord, bool inside,
                 Float distance, Bsdf *bsdf, Medium *int_medium, Medium *ext_medium, Float pdf_area)
        : type_(IntersectionType::kSurface), pos_(pos), normal_(normal), inside_(inside), distance_(distance),
          texcoord_(texcoord), bsdf_(bsdf), medium_(nullptr), int_medium_(int_medium), ext_medium_(ext_medium),
          pdf_area_(pdf_area)
    {
    }

    ///\brief 给定交点处光线出射方向，抽样入射方向和相应的概率等参数
    SamplingRecord Sample(const Vector3 &wo, bool get_attenuation = true) const
    {
        auto rec = SamplingRecord();
        rec.get_attenuation = get_attenuation;
        rec.pos = pos_;
        rec.wo = wo;
        switch (type_)
        {
        case IntersectionType::kSurface:
        {
            if (bsdf_)
            {
                bool one_side = glm::dot(wo, normal_) > 0;  //光线与交点法线是否同侧
                rec.inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
                rec.normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线出射方向夹角小于90度
                rec.texcoord = texcoord_;
                bsdf_->Sample(rec);
            }
            else
            {
                rec.pdf = 1;
                rec.wi = wo;
                rec.type = ScatteringType::kTransimission;
                rec.attenuation = Spectrum(1);
            }
            break;
        }
        case IntersectionType::kMedium:
            medium_->SamplePhaseFunction(rec);
            break;
        default:
            rec.pdf = 1;
            rec.wi = wo;
            rec.type = ScatteringType::kTransimission;
            rec.attenuation = Spectrum(1);
            break;
        }
        return rec;
    }

    ///\brief 根据光线入射方向和出射方向，计算交点处光线传播的 BSDF 系数
    SamplingRecord Eval(const Vector3 &wi, const Vector3 &wo) const
    {
        auto rec = SamplingRecord();
        rec.pos = pos_;
        rec.wi = wi;
        rec.wo = wo;
        switch (type_)
        {
        case IntersectionType::kSurface:
        {
            if (bsdf_)
            {
                bool one_side = glm::dot(wi, normal_) < 0;  //入射光线与法线是否同侧
                rec.inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
                rec.normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线入射方向夹角大于90度
                rec.texcoord = texcoord_;
                bsdf_->Eval(rec);
            }
            else
            {
                rec.pdf = 1;
                rec.type = ScatteringType::kTransimission;
                rec.attenuation = Spectrum(1);
            }
            break;
        }
        case IntersectionType::kMedium:
            medium_->EvalPhaseFunction(rec);
            break;
        default:
            rec.pdf = 1;
            rec.type = ScatteringType::kTransimission;
            rec.attenuation = Spectrum(1);
            break;
        }
        return rec;
    }

    bool SurfaceScattering() const { return type_ == IntersectionType::kSurface; }

    ///\return 交点的位置
    Vector3 pos() const { return pos_; }

    ///\return 交点处的物体表面的法线
    Vector3 normal() const { return normal_; }

    ///\return 光线与物体的相交是否发生
    bool valid() const { return type_ != IntersectionType::kNone; }

    ///\return 光线与单面材质的物体交于物体背面而被吸收
    bool absorb() const { return type_ == IntersectionType::kAbsorbSurface; }

    ///\return 从光线起点到该交点的距离
    Float distance() const { return distance_; }

    ///\return 交点处的物体表面是否发光
    bool HasEmission() const { return bsdf_ && bsdf_->HasEmission(); }

    ///\return 交点处的物体表面的辐射亮度
    Spectrum radiance() const { return bsdf_->radiance(); }

    ///\brief 面元概率
    Float pdf_area() const { return pdf_area_; }

    ///\return 交点处表面材质对应的散射波瓣分布是否是 δ-函数
    bool HarshLobe() const
    {
        return type_ == IntersectionType::kViewPoint ||
               (type_ == IntersectionType::kSurface && (!bsdf_ || bsdf_->HarshLobe()));
    }

    bool Inner(const Vector3 &wo) const
    {
        if (glm::dot(wo, normal_) > 0)
            return inside_;
        else
            return !inside_;
    }

    Medium *medium(const Vector3 &wo) const
    {
        switch (type_)
        {
        case IntersectionType::kSurface:
        {
            bool inside = inside_;
            if (glm::dot(wo, normal_) < 0)
                inside = !inside;

            if (inside)
                return int_medium_;
            else
                return ext_medium_;
            break;
        }
        default:
            return medium_;
            break;
        }
    }

private:
    IntersectionType type_; //光线与物体的相交是否发生
    bool inside_;           //交点处法线是否朝内
    Float distance_;        //从光线起点到该交点的距离
    Float pdf_area_;        //面元概率
    Vector2 texcoord_;      //交点纹理坐标
    Vector3 pos_;           //交点空间坐标
    Vector3 normal_;        //交点法线
    Bsdf *bsdf_;            //交点面片对应的材质
    Medium *medium_;        //物体内部的介质
    Medium *int_medium_;    //物体内部的介质
    Medium *ext_medium_;    //物体内部的介质
};

NAMESPACE_END(raytracer)