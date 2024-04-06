#pragma once

#include "../global.hpp"
#include "../math/math.hpp"

NAMESPACE_BEGIN(raytracer)

//光线散射的类型
enum class ScatteringType
{
    kNone,          //无效
    kScattering,    //散射
    kReflect,       //反射
    kTransimission, //透射
};

//按 BSDF 采样的记录
struct SamplingRecord
{
    ScatteringType type; //记录是否有效
    bool inside;         //表面法线方向是否朝向表面内侧
    double pdf;          //光线传播的概率
    double distance;     //直接抽样光源时从光源到着色点的距离
    dvec2 texcoord;      //表面纹理坐标
    dvec3 normal;        //表面法线方向
    dvec3 tangent;       //表面切线方向
    dvec3 bitangent;     //表面副切线方向
    dvec3 position;      //散射位置
    dvec3 wi;            //入射光线方向
    dvec3 wo;            //出射光线方向
    dvec3 attenuation;   //光能衰减系数
    dvec3 radiance;      //把散射点看作次级光源时，出射辐射亮度的数学期望

    SamplingRecord()
        : type(ScatteringType::kNone),
          inside(false),
          pdf(0),
          position(dvec3(0)),
          tangent(dvec3(0)),
          bitangent(dvec3(0)),
          wi(dvec3(0)),
          wo(dvec3(0)),
          normal(dvec3(0)),
          texcoord(dvec2(0)),
          attenuation(dvec3(1)),
          radiance(dvec3(0)),
          distance(kMaxDouble)
    {
    }
};

NAMESPACE_END(raytracer)