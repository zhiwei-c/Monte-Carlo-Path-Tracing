#pragma once

#include "sampling_record.hpp"
#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

//光线与景物之间交互的类型
enum class IntersectionType
{
    kNone,              //无交互
    kEmitting,          //面光源表面的发光点
    kSurfaceScattering, //光线被景物表面散射
    kMediumScattering,  //光线被介质散射
    kAbsorbing,         //光线被景物表面吸收
};

//光线与景物之间的交互点
class Intersection
{
public:
    Intersection();
    Intersection(const dvec3 &position, Medium *medium);
    Intersection(const std::string &shape_id, double distance, Medium *medium_int, Medium *medium_ext);
    Intersection(const std::string &shape_id, const dvec3 &position, const dvec3 &normal, double pdf_area, Bsdf *bsdf,
                 Medium *medium_int, Medium *medium_ext);
    Intersection(const std::string &shape_id, bool inside, const dvec3 &position, const dvec3 &normal, const dvec3 &tangent,
                 const dvec3 &bitangent, const dvec2 &texcoord, double distance, double pdf_area, Bsdf *bsdf,
                 Medium *medium_int, Medium *medium_ext);

    SamplingRecord Sample(const dvec3 &wo, Sampler *sampler) const;
    SamplingRecord Eval(const dvec3 &wi, const dvec3 &wo) const;
    SamplingRecord PackGeomtryInfo(const dvec3 &dir) const;

    bool IsEmitter() const;
    bool IsHarshLobe() const;
    bool IsAbsorbed() const { return type_ == IntersectionType::kAbsorbing; }
    bool IsValid() const { return type_ != IntersectionType::kNone; }

    Medium* medium(const dvec3& dir) const;
    dvec3 radiance() const;
    dvec3 position() const { return position_; }
    dvec3 normal() const { return normal_; }
    double pdf_area() const { return pdf_area_; }
    double distance() const { return distance_; }
    const std::string &shape_id() const { return shape_id_; }

private:
    IntersectionType type_; //光线与景物之间交互的类型
    bool inside_;           //景物表面的交互点是否位于景物内部
    Bsdf *bsdf_;            //交互点景物表面对应的材质
    Medium *medium_int_;    //交互点景物表面内侧的介质
    Medium *medium_ext_;    //交互点景物表面外侧的介质
    Medium *medium_;        //散射点所处的介质
    double distance_;       //从光线起点到交互点的距离
    double pdf_area_;       //交互点对应的面元概率
    std::string shape_id_;  //物体ID
    dvec2 texcoord_;        //交互点纹理坐标
    dvec3 position_;        //交互点空间坐标
    dvec3 normal_;          //交互点法线
    dvec3 tangent_;         //交互点切线
    dvec3 bitangent_;       //交互点副切线
};

NAMESPACE_END(raytracer)