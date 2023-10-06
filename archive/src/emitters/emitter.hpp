#pragma once

#include "../core/intersection.hpp"
#include "../core/sampling_record.hpp"
#include "../global.hpp"

enum class EmitterType
{
    kArea,        //面光源
    kPoint,       //点光源
    kEnvmap,      //环境光
    kSpot,        //聚光灯
    kDirectional, //遥远的方向光
    kSky,         //天空
    kSun,         //太阳
};

NAMESPACE_BEGIN(raytracer)

//光源
class Emitter
{
public:
    virtual ~Emitter() {}

    virtual SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                                  Accelerator *accelerator) const = 0;
    virtual Intersection SamplePoint(Sampler *sampler) const { return Intersection(); }
    virtual double Pdf(const dvec3 &look_dir) const { return 0; }
    virtual dvec3 radiance(const dvec3 &position, const dvec3 &wi) const { return dvec3(0); }
    EmitterType type() const { return type_; }

protected:
    Emitter(EmitterType type) : type_(type) {}

private:
    EmitterType type_; //光源类型
};

NAMESPACE_END(raytracer)