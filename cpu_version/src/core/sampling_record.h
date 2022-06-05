#pragma once

#include "../utils/math.h"

NAMESPACE_BEGIN(raytracer)
//光线散射类型
enum class ScatteringType
{
    kNone,          //无效
    kReflect,       //反射
    kTransimission, //透射
    kScattering,    //散射
};

//按 BSDF 采样记录
struct SamplingRecord
{
    ScatteringType type;  //记录是否有效
    bool inside;          //表面法线方向是否朝向表面内侧
    bool get_attenuation; //是否计算光能衰减系数
    Float pdf;            //光线传播的概率
    Vector2 texcoord;     //表面纹理坐标
    Vector3 normal;       //表面法线方向
    Vector3 pos;          //散射位置
    Vector3 wi;           //光线入射方向
    Vector3 wo;           //光线出射方向
    Spectrum attenuation; //光能衰减系数

    SamplingRecord()
        : type(ScatteringType::kNone), inside(false), get_attenuation(true), pdf(0), texcoord(Vector2(0)),
          normal(Vector3(0)), pos(Vector3(0)), wi(Vector3(0)), wo(Vector3(0)), attenuation(Spectrum(0))
    {
    }
};

NAMESPACE_END(raytracer)